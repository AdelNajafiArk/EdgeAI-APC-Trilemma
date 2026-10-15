# ================================================================
# APC-EAI V3: ADAPTIVE PARETO-CONSTRAINED EDGE AI
# Local Windows + Quadro P2000 (4 GB VRAM)

# nvidia-smi
# Sun Oct 11 18:00:48 2026
# +-----------------------------------------------------------------------------------------+
# | NVIDIA-SMI 582.78                 Driver Version: 582.78         CUDA Version: 13.0     |
# +-----------------------------------------+------------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |===============================+======================+======================|
# |   0  Quadro P2000        WDDM  | 00000000:01:00.0 Off |                  N/A|
# | N/A   45C    P8    N/A /  N/A |     15MiB /  4096MiB |      0%      Default |
# +-------------------------------+----------------------+----------------------+
# python --version
# Python 3.11
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# PyTorch: 2.4.0+cu121
# CUDA available: True
# Device: Quadro P2000
# pip install medmnist
# pip install opacus
# pip install numpy pandas matplotlib scikit-learn
# ================================================================

import os
import sys
import time
import random
import json
import copy
import math
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms

try:
    import medmnist
    from medmnist import INFO
    HAS_MEDMNIST = True
except ImportError:
    HAS_MEDMNIST = False
    print("[WARN] medmnist not installed")

try:
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
    HAS_OPACUS = True
except ImportError:
    HAS_OPACUS = False
    print("[WARN] opacus not installed")


# ================================================================
# CONFIG
# ================================================================

@dataclass
class Config:
    seed: int = 42
    data_dir: str = "C:/Users/A.C/Desktop/Trilemma/data"
    output_dir: str = "C:/Users/A.C/Desktop/Trilemma/results"

    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0

    dp_epsilons: Dict[str, float] = field(default_factory=lambda: {
        "Nano": 0.1, "Micro": 0.5, "Speed": 1.0,
        "Balanced": 4.0, "Accuracy": 20.0,
    })
    dp_delta: float = 1e-5
    dp_max_grad_norm: float = 1.0
    use_dp_training: bool = False

    latency_warmup: int = 30
    latency_iterations: int = 200

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dataset: str = "pathmnist"
    train_subset: Optional[int] = None

    def __post_init__(self):
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    @property
    def output_path(self) -> str:
        return os.path.join(self.output_dir, f"{self.dataset}_seed{self.seed}")


PROFILE_HINTS = {
    "emergency":    {"acc_percentile": 0.15, "lat_percentile": 0.20, "privacy": 0.10,
                     "description": "Real-time safety-critical, speed dominates"},
    "surveillance": {"acc_percentile": 0.40, "lat_percentile": 0.40, "privacy": 0.40,
                     "description": "Continuous monitoring, moderate accuracy"},
    "monitoring":   {"acc_percentile": 0.60, "lat_percentile": 0.60, "privacy": 0.45,
                     "description": "Industrial monitoring, balanced"},
    "diagnosis":    {"acc_percentile": 0.85, "lat_percentile": 0.80, "privacy": 0.60,
                     "description": "Clinical diagnosis, accuracy + privacy"},
    "telemedicine": {"acc_percentile": 0.75, "lat_percentile": 0.90, "privacy": 0.85,
                     "description": "Remote consultation, extreme privacy"},
    "research":     {"acc_percentile": 0.95, "lat_percentile": 1.00, "privacy": 0.10,
                     "description": "Offline analysis, accuracy dominates"},
}


# ================================================================
# GPU
# ================================================================

def configure_gpu():
    if not torch.cuda.is_available():
        print("[WARN] CUDA not available. Running on CPU.")
        return
    torch.backends.cudnn.benchmark = True
    props = torch.cuda.get_device_properties(0)
    print(f"[GPU] {props.name}")
    print(f"[GPU] Compute capability: {props.major}.{props.minor}")
    print(f"[GPU] Total VRAM: {props.total_memory / 1e9:.2f} GB")


def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ================================================================
# UTIL
# ================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def flatten_labels(y):
    if y.dim() > 1:
        y = y.view(-1)
    return y.long()


# ================================================================
# PRIVACY
# ================================================================

class PrivacyMechanism:
    def __init__(self, epsilon, delta=1e-5, sensitivity=1.0):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        if epsilon > 0:
            self.noise_std = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
        else:
            self.noise_std = 0.0

    def apply(self, x):
        if self.epsilon <= 0 or self.noise_std == 0:
            return x
        return x + torch.randn_like(x) * self.noise_std

    def privacy_score(self):
        if self.epsilon <= 0:
            return 1.0
        return float(1.0 / (1.0 + self.epsilon))


def build_privacy_mechanisms(dp_epsilons, delta=1e-5):
    return {name: PrivacyMechanism(eps, delta) for name, eps in dp_epsilons.items()}


# ================================================================
# DATA
# ================================================================

class FlatLabelDataset(torch.utils.data.Dataset):
    def __init__(self, base):
        self.base = base
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        x, y = self.base[idx]
        y = torch.as_tensor(y).long().view(-1)[0]
        return x, y


def load_pathmnist(cfg):
    if not HAS_MEDMNIST:
        raise ImportError("medmnist required")
    info = INFO["pathmnist"]
    n_classes = len(info["label"])
    DataClass = getattr(medmnist, info["python_class"])

    tf_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    train_set = FlatLabelDataset(DataClass(split="train", root=cfg.data_dir, download=True, transform=tf_train))
    val_set = FlatLabelDataset(DataClass(split="val", root=cfg.data_dir, download=True, transform=tf_test))
    test_set = FlatLabelDataset(DataClass(split="test", root=cfg.data_dir, download=True, transform=tf_test))
    return train_set, val_set, test_set, (3, 64, 64), n_classes


def load_cifar100(cfg):
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    train_full = datasets.CIFAR100(cfg.data_dir, train=True, download=True, transform=tf_train)
    test_set = datasets.CIFAR100(cfg.data_dir, train=False, download=True, transform=tf_test)
    train_set, val_set = random_split(train_full, [45000, 5000],
                                       generator=torch.Generator().manual_seed(cfg.seed))
    return train_set, val_set, test_set, (3, 32, 32), 100


def load_cifar10(cfg):
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_full = datasets.CIFAR10(cfg.data_dir, train=True, download=True, transform=tf_train)
    test_set = datasets.CIFAR10(cfg.data_dir, train=False, download=True, transform=tf_test)
    train_set, val_set = random_split(train_full, [45000, 5000],
                                       generator=torch.Generator().manual_seed(cfg.seed))
    return train_set, val_set, test_set, (3, 32, 32), 10


def load_dataset(cfg):
    loaders = {
        "pathmnist": load_pathmnist,
        "cifar100": load_cifar100,
        "cifar10": load_cifar10,
    }
    train_set, val_set, test_set, input_shape, n_classes = loaders[cfg.dataset](cfg)
    if cfg.train_subset:
        idx = torch.randperm(len(train_set))[:cfg.train_subset]
        train_set = Subset(train_set, idx)

    pin = cfg.device.startswith("cuda")
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=pin)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=pin)
    return train_loader, val_loader, test_loader, input_shape, n_classes


# ================================================================
# MODELS
# ================================================================

class NanoNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(16, n_classes)
    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


class MicroNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(32, n_classes)
    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


class SpeedNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(96, n_classes)
    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


class BalancedNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(),
            nn.Conv2d(48, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(),
            nn.Conv2d(96, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 160, 3, padding=1), nn.BatchNorm2d(160), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(160, n_classes)
    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


class AccuracyNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(256, n_classes)
    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


MODEL_ZOO = {
    "Nano": NanoNet, "Micro": MicroNet, "Speed": SpeedNet,
    "Balanced": BalancedNet, "Accuracy": AccuracyNet,
}


# ================================================================
# TRAIN
# ================================================================

@torch.no_grad()
def evaluate_accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = flatten_labels(y).to(device, non_blocking=True)
        logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total


def train_model(model, train_loader, val_loader, cfg, name, use_dp=False):
    model = model.to(cfg.device)
    dp_active = False
    privacy_engine = None

    if use_dp and HAS_OPACUS:
        try:
            model = ModuleValidator.fix(model)
            privacy_engine = PrivacyEngine()
            optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            target_eps = cfg.dp_epsilons.get(name, 8.0)
            model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                module=model, optimizer=optimizer, data_loader=train_loader,
                epochs=cfg.epochs, target_epsilon=target_eps,
                target_delta=cfg.dp_delta, max_grad_norm=cfg.dp_max_grad_norm,
            )
            dp_active = True
        except Exception as e:
            print(f"  [WARN] DP setup failed: {e}")
            optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.CrossEntropyLoss()

    best_val = 0.0
    best_state = None
    history = []

    for epoch in range(cfg.epochs):
        model.train()
        loss_sum, correct, total = 0.0, 0, 0

        for x, y in train_loader:
            x = x.to(cfg.device, non_blocking=True)
            y = flatten_labels(y).to(cfg.device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

        scheduler.step()
        train_acc = correct / total
        eval_model = model._module if (dp_active and hasattr(model, "_module")) else model
        val_acc = evaluate_accuracy(eval_model, val_loader, cfg.device)

        history.append({"epoch": epoch+1, "train_acc": train_acc,
                        "val_acc": val_acc, "loss": loss_sum/len(train_loader)})

        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(eval_model.state_dict())

        if (epoch + 1) % 5 == 0 or epoch == cfg.epochs - 1:
            eps_str = ""
            if dp_active:
                eps = privacy_engine.get_epsilon(cfg.dp_delta)
                eps_str = f" | ε={eps:.2f}"
            print(f"  {name:10s} | epoch {epoch+1:2d}/{cfg.epochs} | "
                  f"train={train_acc:.3f} | val={val_acc:.3f}{eps_str}")

    eval_model = model._module if (dp_active and hasattr(model, "_module")) else model
    eval_model.load_state_dict(best_state)
    return eval_model, best_val, history


# ================================================================
# LATENCY + CALIBRATION
# ================================================================

@torch.no_grad()
def profile_latency(model, input_shape, cfg):
    model.eval()
    x = torch.randn(1, *input_shape, device=cfg.device)
    for _ in range(cfg.latency_warmup):
        _ = model(x)
    if cfg.device == "cuda":
        torch.cuda.synchronize()

    lats = []
    for _ in range(cfg.latency_iterations):
        if cfg.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(x)
        if cfg.device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        lats.append((t1 - t0) * 1000)

    lats = np.array(lats)
    return {
        "mean_ms": float(np.mean(lats)),
        "median_ms": float(np.median(lats)),
        "p95_ms": float(np.percentile(lats, 95)),
        "p99_ms": float(np.percentile(lats, 99)),
        "std_ms": float(np.std(lats)),
    }


@torch.no_grad()
def compute_calibration_metrics(model, loader, cfg):
    model.eval()
    probs_list, labels_list = [], []
    for x, y in loader:
        x = x.to(cfg.device, non_blocking=True)
        y = flatten_labels(y)
        probs = F.softmax(model(x), dim=1)
        probs_list.append(probs.cpu())
        labels_list.append(y)
    probs = torch.cat(probs_list)
    labels = torch.cat(labels_list)
    conf, pred = probs.max(dim=1)
    correct = pred.eq(labels)

    ece = 0.0
    bounds = torch.linspace(0, 1, 16)
    for i in range(15):
        mask = (conf > bounds[i]) & (conf <= bounds[i+1])
        if mask.any():
            conf_bin = conf[mask].mean().item()
            acc_bin = correct[mask].float().mean().item()
            ece += mask.float().mean().item() * abs(conf_bin - acc_bin)

    nll = F.nll_loss(torch.log(probs + 1e-12), labels).item()
    one_hot = F.one_hot(labels, num_classes=probs.size(1)).float()
    brier = ((probs - one_hot) ** 2).sum(dim=1).mean().item()
    return {"ece": ece, "nll": nll, "brier": brier}


def calibrate_profiles(model_results, profile_hints, privacy_mechanisms):
    accs = sorted([r["test_accuracy"] for r in model_results.values()])
    lats = sorted([r["latency_ms"] for r in model_results.values()])
    acc_min, acc_max = accs[0], accs[-1]
    lat_min, lat_max = lats[0], lats[-1]

    print(f"\n[Calibration] Accuracy range: [{acc_min:.4f}, {acc_max:.4f}]")
    print(f"[Calibration] Latency range: [{lat_min:.3f}, {lat_max:.3f}] ms")

    calibrated = {}
    for name, hint in profile_hints.items():
        base_acc = acc_min + hint["acc_percentile"] * (acc_max - acc_min)
        base_lat = lat_min + hint["lat_percentile"] * (lat_max - lat_min)
        base_priv = hint["privacy"]
        found = False
        best_acc, best_lat = base_acc, base_lat

        for lm in [1.0, 1.1, 1.25, 1.5, 2.0, 2.5, 3.0]:
            for am in [1.0, 0.95, 0.90, 0.85, 0.80]:
                ta, tl = base_acc * am, base_lat * lm
                for mname, r in model_results.items():
                    priv = privacy_mechanisms[mname].privacy_score()
                    if (r["test_accuracy"] >= ta and r["latency_ms"] <= tl and priv >= base_priv):
                        best_acc, best_lat = ta, tl
                        found = True
                        break
                if found: break
            if found: break

        calibrated[name] = {
            "accuracy": best_acc, "privacy": base_priv,
            "latency": best_lat, "description": hint["description"],
            "feasible": found,
        }
        status = "OK" if found else "INFEASIBLE"
        print(f"[Calibration] {name:12s}: acc>={best_acc:.4f}, "
              f"lat<={best_lat:.3f}ms, priv>={base_priv:.2f}  {status}")

    return calibrated


class ConstraintAwareSelector:
    def __init__(self, model_results, privacy_mechanisms):
        self.mr = model_results
        self.pm = privacy_mechanisms

    def satisfies(self, name, profile):
        r = self.mr[name]
        return (r["test_accuracy"] >= profile["accuracy"] and
                self.pm[name].privacy_score() >= profile["privacy"] and
                r["latency_ms"] <= profile["latency"])

    def utility(self, name, profile):
        r = self.mr[name]
        priv = self.pm[name].privacy_score()
        lat_norm = 1.0 - min(r["latency_ms"] / max(profile["latency"], 1e-9), 1.0)
        return 0.5 * r["test_accuracy"] + 0.3 * priv + 0.2 * lat_norm

    def violation(self, name, profile):
        r = self.mr[name]
        priv = self.pm[name].privacy_score()
        acc_gap = max(0, profile["accuracy"] - r["test_accuracy"])
        priv_gap = max(0, profile["privacy"] - priv)
        lat_gap = max(0, r["latency_ms"] - profile["latency"])
        return 2.0 * acc_gap + priv_gap + lat_gap / max(profile["latency"], 1e-9)

    def select(self, profile):
        feasible = [m for m in self.mr if self.satisfies(m, profile)]
        if feasible:
            return max(feasible, key=lambda m: self.utility(m, profile)), True, feasible
        return min(self.mr, key=lambda m: self.violation(m, profile)), False, []


class BaselineSelector:
    def __init__(self, names, seed=42):
        self.names = names
        self.rng = np.random.RandomState(seed)

    def select(self, strategy, profile):
        if strategy == "always_nano": return "Nano"
        if strategy == "always_micro": return "Micro"
        if strategy == "always_speed": return "Speed"
        if strategy == "always_balanced": return "Balanced"
        if strategy == "always_accuracy": return "Accuracy"
        if strategy == "random": return self.rng.choice(self.names)
        if strategy == "threshold":
            lat = profile["latency"]
            if lat < 0.4: return "Nano"
            if lat < 0.6: return "Micro"
            if lat < 1.0: return "Speed"
            if lat < 5.0: return "Balanced"
            return "Accuracy"
        raise ValueError(strategy)


def is_dominated(a, b):
    ge_all = (a["accuracy"] >= b["accuracy"] and
              a["privacy_score"] >= b["privacy_score"] and
              a["latency_ms"] <= b["latency_ms"])
    gt_one = (a["accuracy"] > b["accuracy"] or
              a["privacy_score"] > b["privacy_score"] or
              a["latency_ms"] < b["latency_ms"])
    return ge_all and gt_one


def compute_pareto_front(points):
    return [p for i, p in enumerate(points)
            if not any(is_dominated(q, p) for j, q in enumerate(points) if i != j)]


# ================================================================
# RUN SINGLE SEED
# ================================================================

def run_single_seed(cfg, verbose=True):
    set_seed(cfg.seed)
    if verbose:
        print(f"\n{'='*70}")
        print(f"SEED {cfg.seed} | Dataset: {cfg.dataset} | Device: {cfg.device}")
        print(f"{'='*70}")

    os.makedirs(cfg.output_path, exist_ok=True)
    train_loader, val_loader, test_loader, input_shape, n_classes = load_dataset(cfg)

    if verbose:
        print(f"Train batches: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
        print(f"Input: {input_shape}, Classes: {n_classes}")

    trained_models, val_accs, histories = {}, {}, {}

    for name, ModelCls in MODEL_ZOO.items():
        if verbose:
            print(f"\n[{name}] Training...")
        clear_gpu_cache()
        model = ModelCls(n_classes=n_classes)
        n_params = sum(p.numel() for p in model.parameters())

        trained, best_val, hist = train_model(model, train_loader, val_loader, cfg, name, use_dp=cfg.use_dp_training)
        trained_models[name] = trained
        val_accs[name] = best_val
        histories[name] = hist
        clear_gpu_cache()

        if verbose:
            print(f"  OK {name}: val={best_val:.4f}, params={n_params:,}")

    if verbose:
        print(f"\nEvaluating on test set...")

    model_results = {}
    for name, model in trained_models.items():
        test_acc = evaluate_accuracy(model, test_loader, cfg.device)
        latency = profile_latency(model, input_shape, cfg)
        calibration = compute_calibration_metrics(model, test_loader, cfg)
        n_params = sum(p.numel() for p in model.parameters())

        model_results[name] = {
            "name": name, "val_accuracy": val_accs[name], "test_accuracy": test_acc,
            "latency_ms": latency["mean_ms"], "latency_p95_ms": latency["p95_ms"],
            "latency_p99_ms": latency["p99_ms"], "ece": calibration["ece"],
            "nll": calibration["nll"], "brier": calibration["brier"], "params": n_params,
        }
        if verbose:
            print(f"  {name:10s}: acc={test_acc:.4f}, lat={latency['mean_ms']:.3f}ms, ECE={calibration['ece']:.4f}")

    privacy_mechanisms = build_privacy_mechanisms(cfg.dp_epsilons, cfg.dp_delta)
    for name in model_results:
        model_results[name]["privacy_score"] = privacy_mechanisms[name].privacy_score()
        model_results[name]["epsilon"] = cfg.dp_epsilons[name]

    profiles = calibrate_profiles(model_results, PROFILE_HINTS, privacy_mechanisms)
    selector = ConstraintAwareSelector(model_results, privacy_mechanisms)
    baseline_sel = BaselineSelector(list(MODEL_ZOO.keys()), seed=cfg.seed)

    baseline_strategies = ["always_nano", "always_micro", "always_speed",
                            "always_balanced", "always_accuracy", "random", "threshold"]

    profile_results = []
    for pname, profile in profiles.items():
        selected_model, feasible, _ = selector.select(profile)
        sel_perf = model_results[selected_model]

        baselines = {}
        for strat in baseline_strategies:
            bl_model = baseline_sel.select(strat, profile)
            bl_perf = model_results[bl_model]
            bl_ok = (bl_perf["test_accuracy"] >= profile["accuracy"] and
                     bl_perf["privacy_score"] >= profile["privacy"] and
                     bl_perf["latency_ms"] <= profile["latency"])
            baselines[strat] = {
                "model": bl_model, "satisfied": bl_ok,
                "accuracy": bl_perf["test_accuracy"],
                "latency_ms": bl_perf["latency_ms"],
            }

        profile_results.append({
            "profile": pname, "requirements": profile,
            "selected_model": selected_model,
            "selected_accuracy": sel_perf["test_accuracy"],
            "selected_latency_ms": sel_perf["latency_ms"],
            "selected_privacy_score": sel_perf["privacy_score"],
            "satisfied": feasible, "baselines": baselines,
        })
        if verbose:
            status = "OK" if feasible else "XX"
            print(f"  [{status}] {pname:15s} -> {selected_model:10s} "
                  f"(acc={sel_perf['test_accuracy']:.3f}, lat={sel_perf['latency_ms']:.3f}, "
                  f"priv={sel_perf['privacy_score']:.3f})")

    pareto_candidates = [{"model": n, "accuracy": r["test_accuracy"],
                           "privacy_score": r["privacy_score"],
                           "latency_ms": r["latency_ms"]}
                          for n, r in model_results.items()]
    pareto_front = compute_pareto_front(pareto_candidates)

    result = {
        "seed": cfg.seed, "dataset": cfg.dataset, "device": cfg.device,
        "n_classes": n_classes, "input_shape": list(input_shape),
        "model_results": model_results, "profile_results": profile_results,
        "pareto_front": pareto_front, "training_histories": histories,
        "calibrated_profiles": profiles,
        "privacy_config": {"dp_epsilons": cfg.dp_epsilons, "dp_delta": cfg.dp_delta},
    }
    with open(os.path.join(cfg.output_path, "results.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result


def aggregate_seeds(all_results):
    if not all_results:
        return {}
    model_names = list(all_results[0]["model_results"].keys())
    model_agg = {}
    for name in model_names:
        accs = [r["model_results"][name]["test_accuracy"] for r in all_results]
        lats = [r["model_results"][name]["latency_ms"] for r in all_results]
        eces = [r["model_results"][name]["ece"] for r in all_results]
        model_agg[name] = {
            "test_accuracy_mean": float(np.mean(accs)),
            "test_accuracy_std": float(np.std(accs)),
            "latency_ms_mean": float(np.mean(lats)),
            "latency_ms_std": float(np.std(lats)),
            "ece_mean": float(np.mean(eces)),
            "ece_std": float(np.std(eces)),
            "n_seeds": len(all_results),
        }

    profile_names = [p["profile"] for p in all_results[0]["profile_results"]]
    profile_agg = {}
    for pname in profile_names:
        selections, satisfied = [], []
        baselines = defaultdict(lambda: {"satisfied": []})
        for r in all_results:
            for pr in r["profile_results"]:
                if pr["profile"] == pname:
                    selections.append(pr["selected_model"])
                    satisfied.append(pr["satisfied"])
                    for strat, bl in pr["baselines"].items():
                        baselines[strat]["satisfied"].append(bl["satisfied"])
        selection_dist = {m: selections.count(m)/len(selections) for m in set(selections)}
        profile_agg[pname] = {
            "adaptive_satisfaction_rate": float(np.mean(satisfied)),
            "selection_distribution": selection_dist,
            "baseline_satisfaction_rates": {s: float(np.mean(v["satisfied"])) for s, v in baselines.items()},
            "n_observations": len(satisfied),
        }
    return {"model_aggregate": model_agg, "profile_aggregate": profile_agg, "n_seeds": len(all_results)}


def plot_all_figures(agg, output_dir, dataset):
    os.makedirs(output_dir, exist_ok=True)
    model_agg = agg["model_aggregate"]
    profile_agg = agg["profile_aggregate"]

    # Fig 1
    fig, ax = plt.subplots(figsize=(9, 6))
    names = list(model_agg.keys())
    accs = [model_agg[n]["test_accuracy_mean"] for n in names]
    lats = [model_agg[n]["latency_ms_mean"] for n in names]
    ax.errorbar(lats, accs, fmt='o', markersize=12, color='tab:blue')
    for n, l, a in zip(names, lats, accs):
        ax.annotate(n, (l, a), xytext=(7, 7), textcoords='offset points', fontsize=10)
    ax.set_xlabel("Latency (ms)"); ax.set_ylabel("Test Accuracy")
    ax.set_title(f"Accuracy-Latency Trade-off ({dataset.upper()})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig1_accuracy_latency.png"), dpi=200)
    plt.close()

    # Fig 2
    fig, ax = plt.subplots(figsize=(12, 6))
    profiles = list(profile_agg.keys())
    adaptive = [profile_agg[p]["adaptive_satisfaction_rate"] for p in profiles]
    baselines_list = ["always_nano", "always_speed", "always_balanced",
                      "always_accuracy", "random", "threshold"]
    x = np.arange(len(profiles)); width = 0.11
    ax.bar(x - 3.5*width, adaptive, width, label="Adaptive (ours)", color='tab:red')
    for i, bl in enumerate(baselines_list):
        rates = [profile_agg[p]["baseline_satisfaction_rates"].get(bl, 0) for p in profiles]
        ax.bar(x + (i - 2.5)*width, rates, width, label=bl)
    ax.set_xticks(x); ax.set_xticklabels(profiles, rotation=30, ha='right')
    ax.set_ylabel("Satisfaction Rate"); ax.set_ylim(0, 1.05)
    ax.set_title(f"Satisfaction by Profile ({dataset.upper()})")
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig2_satisfaction.png"), dpi=200)
    plt.close()

    # Fig 3
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.cm.tab10
    bottom = np.zeros(len(profiles))
    for i, m in enumerate(model_agg.keys()):
        dists = [profile_agg[p]["selection_distribution"].get(m, 0) for p in profiles]
        ax.bar(profiles, dists, bottom=bottom, label=m, color=cmap(i))
        bottom += np.array(dists)
    ax.set_ylabel("Selection Probability")
    ax.set_title(f"Adaptive Model Selection ({dataset.upper()})")
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.xticks(rotation=30, ha='right'); ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig3_selection.png"), dpi=200)
    plt.close()
    print(f"\n[INFO] Figures saved to {output_dir}")


def main(dataset="pathmnist", seeds=1, epochs=30, train_subset=None, use_dp_training=False):
    print("=" * 80)
    print("APC-EAI V3: ADAPTIVE PARETO-CONSTRAINED EDGE AI")
    print("=" * 80)
    configure_gpu()

    all_results = []
    for seed_idx in range(seeds):
        cfg = Config(seed=42 + seed_idx, dataset=dataset, epochs=epochs,
                     train_subset=train_subset, use_dp_training=use_dp_training)
        result = run_single_seed(cfg, verbose=True)
        all_results.append(result)

    print(f"\n{'='*80}")
    print("AGGREGATING ACROSS SEEDS")
    print(f"{'='*80}")
    agg = aggregate_seeds(all_results)

    output_root = os.path.join(Config().output_dir, f"{dataset}_aggregate")
    os.makedirs(output_root, exist_ok=True)
    with open(os.path.join(output_root, "aggregate.json"), "w") as f:
        json.dump(agg, f, indent=2, default=str)

    print("\n--- Model Performance ---")
    for name, m in agg["model_aggregate"].items():
        params = all_results[0]["model_results"][name]["params"]
        print(f"{name:<12} acc={m['test_accuracy_mean']:.4f}±{m['test_accuracy_std']:.4f}  "
              f"lat={m['latency_ms_mean']:.3f}±{m['latency_ms_std']:.3f}ms  "
              f"ECE={m['ece_mean']:.4f}  params={params:,}")

    print("\n--- Profile Satisfaction ---")
    baseline_keys = ["always_nano", "always_speed", "always_balanced",
                     "always_accuracy", "random", "threshold"]

    # Short labels that don't rely on split('_')
    short_labels = {
        "always_nano":     "nano",
        "always_micro":    "micr",
        "always_speed":    "sped",
        "always_balanced": "bala",
        "always_accuracy": "accu",
        "random":          "rand",
        "threshold":       "thrs",
    }

    for pname, p in agg["profile_aggregate"].items():
        row = f"{pname:<15} adaptive={p['adaptive_satisfaction_rate']:.3f}  "
        for bl in baseline_keys:
            label = short_labels.get(bl, bl[:4])
            rate = p['baseline_satisfaction_rates'].get(bl, 0.0)
            row += f" {label}={rate:.3f}"
        print(row)

    plot_all_figures(agg, output_root, dataset)
    print(f"\n[DONE] Results: {output_root}")
    return agg


# ================================================================
# ENTRY POINT (REQUIRED for Windows!)
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="pathmnist", choices=["pathmnist", "cifar100", "cifar10"])
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--train_subset", type=int, default=None)
    parser.add_argument("--use_dp", action="store_true")
    args = parser.parse_args()

    main(dataset=args.dataset, seeds=args.seeds, epochs=args.epochs,
         train_subset=args.train_subset, use_dp_training=args.use_dp)