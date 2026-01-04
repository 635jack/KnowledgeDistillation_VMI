"""
Knowledge Distillation Utilities for VMI Lab
=============================================
Shared components for KD notebooks on CIFAR-10
ResNet50 Teacher → ResNet18 Student

Author:Jacques Gastebois
Seed: 42
"""

import os
import json
import random
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm

# ============================================================================
# CONFIGURATION
# ============================================================================

SEED = 42
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)
CIFAR_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck']

# Solarized theme colors
SOLARIZED = {
    'base03': '#002b36',
    'base02': '#073642',
    'base01': '#586e75',
    'base00': '#657b83',
    'base0': '#839496',
    'base1': '#93a1a1',
    'base2': '#eee8d5',
    'base3': '#fdf6e3',
    'yellow': '#b58900',
    'orange': '#cb4b16',
    'red': '#dc322f',
    'magenta': '#d33682',
    'violet': '#6c71c4',
    'blue': '#268bd2',
    'cyan': '#2aa198',
    'green': '#859900',
}


def set_seed(seed: int = SEED):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ============================================================================
# CONFIGURATION DATACLASS
# ============================================================================

@dataclass
class KDConfig:
    """Configuration for Knowledge Distillation experiments"""
    # Data
    data_root: str = "./data"
    batch_size: int = 128
    num_workers: int = 4
    val_ratio: float = 0.1
    
    # Training
    epochs: int = 50
    lr: float = 0.01
    weight_decay: float = 1e-4
    lr_step: int = 30
    lr_gamma: float = 0.2
    
    # KD parameters
    temperature: float = 4.0
    alpha: float = 0.9
    beta: float = 0.5  # Feature distillation weight
    
    # Misc
    seed: int = SEED
    precision: str = "amp"
    
    def to_dict(self):
        return asdict(self)


# ============================================================================
# DATA LOADING
# ============================================================================

def get_cifar10_loaders(cfg: KDConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create CIFAR-10 train/val/test loaders"""
    set_seed(cfg.seed)
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])
    
    full_train = torchvision.datasets.CIFAR10(
        root=cfg.data_root, train=True, download=True, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root=cfg.data_root, train=False, download=True, transform=test_transform
    )
    
    val_len = int(len(full_train) * cfg.val_ratio)
    train_len = len(full_train) - val_len
    train_set, val_set = random_split(
        full_train, [train_len, val_len],
        generator=torch.Generator().manual_seed(cfg.seed)
    )
    
    loader_kwargs = {
        'batch_size': cfg.batch_size,
        'num_workers': cfg.num_workers,
        'pin_memory': True
    }
    
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)
    
    return train_loader, val_loader, test_loader


def get_sample_images(test_loader: DataLoader, n: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get n sample images from test set for visualization"""
    images, labels = next(iter(test_loader))
    return images[:n], labels[:n]


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Denormalize CIFAR-10 images for visualization"""
    mean = torch.tensor(CIFAR_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR_STD).view(3, 1, 1)
    return tensor * std + mean


# ============================================================================
# MODEL BUILDERS
# ============================================================================

def build_resnet(version: str, num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    """Build ResNet model for CIFAR-10"""
    weights = None
    if pretrained:
        if version == "resnet50":
            weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        elif version == "resnet18":
            weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    
    if version == "resnet50":
        model = torchvision.models.resnet50(weights=weights)
    elif version == "resnet18":
        model = torchvision.models.resnet18(weights=weights)
    else:
        raise ValueError(f"Unknown ResNet version: {version}")
    
    # Replace final FC for CIFAR-10
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model


def build_mobilenet_v3(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    """Build MobileNetV3 Small for CIFAR-10"""
    weights = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = torchvision.models.mobilenet_v3_small(weights=weights)
    
    # Replace classifier
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    
    return model


class TinyConvNet(nn.Module):
    """Custom small CNN for bonus experiments"""
    
    def __init__(self, base_width: int = 64, num_blocks: int = 3, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_width, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True)
        )
        
        width = base_width
        stages = []
        for idx in range(num_blocks):
            out_width = width * 2 if idx < num_blocks - 1 else width
            stride = 2 if idx < num_blocks - 1 else 1
            stages.append(self._make_stage(width, out_width, stride))
            if idx < num_blocks - 1:
                width *= 2
        
        self.blocks = nn.Sequential(*stages)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width, num_classes)
        )
        
        # Store intermediate feature layer name for hooks
        self.feature_layer_name = "blocks.1"
    
    def _make_stage(self, in_c: int, out_c: int, stride: int):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


def count_params(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model: nn.Module, name: str):
    """Print model parameter info"""
    params = count_params(model)
    print(f"📊 {name}: {params:,} parameters ({params/1e6:.2f}M)")


# ============================================================================
# FEATURE HOOKS & ADAPTERS
# ============================================================================

class FeatureHook:
    """Hook to capture intermediate features during forward pass"""
    
    def __init__(self, module: nn.Module):
        self.features = None
        self.handle = module.register_forward_hook(self._hook_fn)
    
    def _hook_fn(self, module, input, output):
        self.features = output
    
    def close(self):
        self.handle.remove()


def attach_hooks(model: nn.Module, layer_names: List[str]) -> Dict[str, FeatureHook]:
    """Attach hooks to specified layers"""
    named_modules = dict(model.named_modules())
    hooks = {}
    
    for name in layer_names:
        if name not in named_modules:
            raise ValueError(f"Layer '{name}' not found in model. Available: {list(named_modules.keys())[:20]}...")
        hooks[name] = FeatureHook(named_modules[name])
    
    return hooks


def get_hook_features(hooks: Dict[str, FeatureHook]) -> Dict[str, torch.Tensor]:
    """Get features from all hooks"""
    return {name: hook.features for name, hook in hooks.items()}


def release_hooks(hooks: Dict[str, FeatureHook]):
    """Release all hooks"""
    for hook in hooks.values():
        hook.close()


class FeatureAdapter(nn.Module):
    """Adapt feature dimensions between teacher and student"""
    
    def __init__(self, student_channels: int, teacher_channels: int):
        super().__init__()
        self.adapter = nn.Conv2d(student_channels, teacher_channels, kernel_size=1)
    
    def forward(self, x):
        return self.adapter(x)


# ============================================================================
# KNOWLEDGE DISTILLATION LOSSES
# ============================================================================

def kd_loss_mse(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                temperature: float = 4.0) -> torch.Tensor:
    """MSE loss on scaled logits"""
    return F.mse_loss(student_logits / temperature, teacher_logits / temperature)


def kd_loss_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
               temperature: float = 4.0) -> torch.Tensor:
    """KL divergence loss on soft probabilities"""
    p = F.log_softmax(student_logits / temperature, dim=1)
    q = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(p, q, reduction='batchmean') * (temperature ** 2)


def feature_loss_mse(student_features: Dict[str, torch.Tensor],
                     teacher_features: Dict[str, torch.Tensor],
                     adapters: Optional[Dict[str, FeatureAdapter]] = None) -> torch.Tensor:
    """MSE loss on intermediate features"""
    total_loss = 0.0
    
    for name in student_features:
        s_feat = student_features[name]
        t_feat = teacher_features[name]
        
        # Apply adapter if dimensions don't match
        if adapters and name in adapters:
            s_feat = adapters[name](s_feat)
        
        # Ensure same spatial dimensions
        if s_feat.shape[-2:] != t_feat.shape[-2:]:
            s_feat = F.adaptive_avg_pool2d(s_feat, t_feat.shape[-2:])
        
        total_loss += F.mse_loss(s_feat, t_feat)
    
    return total_loss / len(student_features)


def feature_loss_l1(student_features: Dict[str, torch.Tensor],
                    teacher_features: Dict[str, torch.Tensor],
                    adapters: Optional[Dict[str, FeatureAdapter]] = None) -> torch.Tensor:
    """L1 loss on intermediate features"""
    total_loss = 0.0
    
    for name in student_features:
        s_feat = student_features[name]
        t_feat = teacher_features[name]
        
        # Apply adapter if dimensions don't match
        if adapters and name in adapters:
            s_feat = adapters[name](s_feat)
        
        # Ensure same spatial dimensions
        if s_feat.shape[-2:] != t_feat.shape[-2:]:
            s_feat = F.adaptive_avg_pool2d(s_feat, t_feat.shape[-2:])
        
        total_loss += F.l1_loss(s_feat, t_feat)
    
    return total_loss / len(student_features)


def feature_loss_cosine(student_features: Dict[str, torch.Tensor],
                        teacher_features: Dict[str, torch.Tensor],
                        adapters: Optional[Dict[str, FeatureAdapter]] = None) -> torch.Tensor:
    """Cosine similarity loss on intermediate features"""
    total_loss = 0.0
    
    for name in student_features:
        s_feat = student_features[name].flatten(1)
        t_feat = teacher_features[name].flatten(1)
        
        if adapters and name in adapters:
            s_feat = adapters[name](student_features[name]).flatten(1)
        
        cos_sim = F.cosine_similarity(s_feat, t_feat, dim=1)
        total_loss += (1 - cos_sim.mean())
    
    return total_loss / len(student_features)


def skd_affinity_loss(student_features: Dict[str, torch.Tensor],
                      teacher_features: Dict[str, torch.Tensor]) -> torch.Tensor:
    """SKD-like affinity distillation loss"""
    total_loss = 0.0
    
    for name in student_features:
        s = student_features[name].flatten(2)  # B x C x HW
        t = teacher_features[name].flatten(2)
        
        # Compute affinity matrices
        s_affinity = torch.bmm(s.transpose(1, 2), s)  # B x HW x HW
        t_affinity = torch.bmm(t.transpose(1, 2), t)
        
        # Normalize
        s_affinity = F.normalize(s_affinity, dim=-1)
        t_affinity = F.normalize(t_affinity, dim=-1)
        
        total_loss += F.mse_loss(s_affinity, t_affinity)
    
    return total_loss / len(student_features)


# ============================================================================
# GRADCAM VISUALIZATION
# ============================================================================

class GradCAM:
    """Gradient-weighted Class Activation Mapping"""
    
    def __init__(self, model: nn.Module, target_layer: str):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        modules = dict(model.named_modules())
        if target_layer not in modules:
            raise ValueError(f"Layer {target_layer} not found")
        
        target = modules[target_layer]
        target.register_forward_hook(self._forward_hook)
        target.register_full_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def __call__(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """Generate GradCAM heatmap"""
        self.model.eval()
        
        # Enable gradients for input tensor
        input_tensor = input_tensor.clone().requires_grad_(True)
        
        # Forward pass with gradients enabled
        with torch.enable_grad():
            output = self.model(input_tensor)
            
            if class_idx is None:
                class_idx = output.argmax(dim=1)
            
            # Backward pass
            self.model.zero_grad()
            one_hot = torch.zeros_like(output)
            one_hot[0, class_idx] = 1
            output.backward(gradient=one_hot, retain_graph=True)
        
        # Compute CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode='bilinear', align_corners=False)
        
        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam


def plot_gradcam_comparison(images: torch.Tensor, labels: torch.Tensor,
                            teacher: nn.Module, student: nn.Module,
                            teacher_layer: str = "layer4",
                            student_layer: str = "layer4",
                            device: torch.device = None,
                            save_path: Optional[str] = None):
    """Compare GradCAM heatmaps between teacher and student"""
    if device is None:
        device = get_device()
    
    teacher = teacher.to(device).eval()
    student = student.to(device).eval()
    images = images.to(device)
    
    n_images = len(images)
    fig, axes = plt.subplots(n_images, 4, figsize=(16, 4 * n_images))
    
    # Apply Solarized style
    fig.patch.set_facecolor(SOLARIZED['base3'])
    
    teacher_cam = GradCAM(teacher, teacher_layer)
    student_cam = GradCAM(student, student_layer)
    
    for i in range(n_images):
        img = images[i:i+1]
        label = labels[i].item()
        
        # Original image
        img_display = denormalize(img[0].cpu()).permute(1, 2, 0).numpy()
        img_display = np.clip(img_display, 0, 1)
        
        # GradCAM heatmaps
        t_cam = teacher_cam(img, label)
        s_cam = student_cam(img, label)
        
        # Plot
        ax_row = axes[i] if n_images > 1 else axes
        
        ax_row[0].imshow(img_display)
        ax_row[0].set_title(f"Original\n{CIFAR_CLASSES[label]}", fontsize=10,
                           color=SOLARIZED['base01'])
        ax_row[0].axis('off')
        
        ax_row[1].imshow(img_display)
        ax_row[1].imshow(t_cam, cmap='jet', alpha=0.5)
        ax_row[1].set_title("Teacher GradCAM", fontsize=10, color=SOLARIZED['blue'])
        ax_row[1].axis('off')
        
        ax_row[2].imshow(img_display)
        ax_row[2].imshow(s_cam, cmap='jet', alpha=0.5)
        ax_row[2].set_title("Student GradCAM", fontsize=10, color=SOLARIZED['green'])
        ax_row[2].axis('off')
        
        # Difference
        diff = np.abs(t_cam - s_cam)
        ax_row[3].imshow(diff, cmap='hot')
        ax_row[3].set_title("Difference", fontsize=10, color=SOLARIZED['red'])
        ax_row[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                   facecolor=SOLARIZED['base3'])
    
    plt.show()


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

class TrainingHistory:
    """Track training metrics"""
    
    def __init__(self):
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'test_loss': None, 'test_acc': None
        }
    
    def update(self, phase: str, loss: float, acc: float):
        if phase in ['train', 'val']:
            self.history[f'{phase}_loss'].append(loss)
            self.history[f'{phase}_acc'].append(acc)
        elif phase == 'test':
            self.history['test_loss'] = loss
            self.history['test_acc'] = acc
    
    def get_best_val_acc(self) -> Tuple[float, int]:
        if not self.history['val_acc']:
            return 0.0, 0
        best_acc = max(self.history['val_acc'])
        best_epoch = self.history['val_acc'].index(best_acc)
        return best_acc, best_epoch


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Evaluate model on a data loader"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    return {
        'loss': total_loss / total,
        'acc': correct / total
    }


def train_epoch(model: nn.Module, loader: DataLoader, optimizer, 
                criterion, device: torch.device, scaler=None,
                teacher: Optional[nn.Module] = None,
                kd_config: Optional[dict] = None,
                adapters: Optional[Dict[str, FeatureAdapter]] = None,
                student_hooks: Optional[Dict] = None,
                teacher_hooks: Optional[Dict] = None) -> Dict[str, float]:
    """Train for one epoch with optional KD"""
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Use AMP if available
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            outputs = model(images)
            ce_loss = criterion(outputs, labels)
            
            loss = ce_loss
            
            # Knowledge Distillation
            if teacher is not None and kd_config is not None:
                teacher.eval()
                with torch.no_grad():
                    teacher_outputs = teacher(images)
                
                # Score distillation (logits)
                alpha = kd_config.get('alpha', 0.9)
                temperature = kd_config.get('temperature', 4.0)
                
                if kd_config.get('loss_type', 'mse') == 'kl':
                    kd_loss = kd_loss_kl(outputs, teacher_outputs, temperature)
                else:
                    kd_loss = kd_loss_mse(outputs, teacher_outputs, temperature)
                
                # Feature distillation
                if kd_config.get('use_features', False) and student_hooks and teacher_hooks:
                    student_feats = get_hook_features(student_hooks)
                    teacher_feats = get_hook_features(teacher_hooks)
                    
                    beta = kd_config.get('beta', 0.5)
                    feat_metric = kd_config.get('feature_metric', 'mse')
                    
                    if feat_metric == 'cosine':
                        feat_loss = feature_loss_cosine(student_feats, teacher_feats, adapters)
                    elif feat_metric == 'l1':
                        feat_loss = feature_loss_l1(student_feats, teacher_feats, adapters)
                    elif feat_metric == 'skd':
                        feat_loss = skd_affinity_loss(student_feats, teacher_feats)
                    else:
                        feat_loss = feature_loss_mse(student_feats, teacher_feats, adapters)
                    
                    kd_loss = kd_loss + beta * feat_loss
                
                # Combined loss
                loss = (1 - alpha) * ce_loss + alpha * kd_loss
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # Track metrics
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    return {
        'loss': total_loss / total,
        'acc': correct / total
    }


# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

def setup_solarized_style():
    """Configure matplotlib for Solarized theme"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.facecolor': SOLARIZED['base3'],
        'axes.facecolor': SOLARIZED['base3'],
        'axes.edgecolor': SOLARIZED['base01'],
        'axes.labelcolor': SOLARIZED['base01'],
        'text.color': SOLARIZED['base01'],
        'xtick.color': SOLARIZED['base01'],
        'ytick.color': SOLARIZED['base01'],
        'grid.color': SOLARIZED['base2'],
        'figure.figsize': (12, 6),
        'font.size': 11
    })


def plot_training_curves(histories: Dict[str, TrainingHistory],
                        title: str = "Training Curves",
                        save_path: Optional[str] = None):
    """Plot loss and accuracy curves for multiple models"""
    setup_solarized_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = [SOLARIZED['blue'], SOLARIZED['green'], SOLARIZED['orange'],
              SOLARIZED['magenta'], SOLARIZED['cyan']]
    
    for idx, (name, history) in enumerate(histories.items()):
        color = colors[idx % len(colors)]
        epochs = range(1, len(history.history['train_loss']) + 1)
        
        # Loss subplot
        axes[0].plot(epochs, history.history['train_loss'], '-',
                    color=color, alpha=0.7, label=f'{name} (train)')
        axes[0].plot(epochs, history.history['val_loss'], '--',
                    color=color, label=f'{name} (val)')
        
        # Accuracy subplot
        axes[1].plot(epochs, history.history['train_acc'], '-',
                    color=color, alpha=0.7, label=f'{name} (train)')
        axes[1].plot(epochs, history.history['val_acc'], '--',
                    color=color, label=f'{name} (val)')
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].legend()
    
    fig.suptitle(title, fontsize=14, color=SOLARIZED['base01'])
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                   facecolor=SOLARIZED['base3'])
    plt.show()


def plot_confusion_matrix(model: nn.Module, loader: DataLoader,
                         device: torch.device, title: str = "Confusion Matrix",
                         save_path: Optional[str] = None):
    """Plot confusion matrix"""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    
    setup_solarized_style()
    plt.close('all')
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CIFAR_CLASSES, yticklabels=CIFAR_CLASSES,
                ax=ax)
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_logits_distribution(model: nn.Module, loader: DataLoader,
                            device: torch.device, temperature: float = 4.0,
                            title: str = "Soft Logits Distribution",
                            save_path: Optional[str] = None):
    """Plot distribution of soft logits"""
    model.eval()
    all_soft_logits = []
    
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            outputs = model(images)
            soft = F.softmax(outputs / temperature, dim=1)
            all_soft_logits.append(soft.cpu())
            
            if len(all_soft_logits) > 10:  # Limit samples
                break
    
    soft_logits = torch.cat(all_soft_logits, dim=0).numpy()
    
    setup_solarized_style()
    plt.close('all')
    fig, ax = plt.subplots(figsize=(12, 5))
    
    for i, class_name in enumerate(CIFAR_CLASSES):
        ax.hist(soft_logits[:, i], bins=50, alpha=0.5, label=class_name)
    
    ax.set_xlabel('Soft Probability')
    ax.set_ylabel('Count')
    ax.set_title(f'{title} (T={temperature})')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================================
# TENSORBOARD UTILITIES
# ============================================================================

class TBLogger:
    """TensorBoard logger wrapper"""
    
    def __init__(self, log_dir: str = "runs", experiment_name: str = "kd_experiment"):
        self.log_dir = Path(log_dir) / experiment_name
        self.writer = SummaryWriter(self.log_dir)
        print(f"📊 TensorBoard logs: {self.log_dir}")
        print(f"   Run: tensorboard --logdir={log_dir}")
    
    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_values: Dict[str, float], step: int):
        self.writer.add_scalars(main_tag, tag_values, step)
    
    def log_histogram(self, tag: str, values: torch.Tensor, step: int):
        self.writer.add_histogram(tag, values, step)
    
    def log_embedding(self, features: torch.Tensor, labels: List[str],
                     tag: str = "embeddings", step: int = 0):
        self.writer.add_embedding(features, metadata=labels, tag=tag, global_step=step)
    
    def log_image(self, tag: str, image: torch.Tensor, step: int):
        self.writer.add_image(tag, image, step)
    
    def log_model_graph(self, model: nn.Module, input_tensor: torch.Tensor):
        self.writer.add_graph(model, input_tensor)
    
    def close(self):
        self.writer.close()


def log_embeddings_comparison(teacher: nn.Module, student: nn.Module,
                              loader: DataLoader, device: torch.device,
                              tb_logger: TBLogger, epoch: int = 0):
    """Log embedding comparison to TensorBoard Projector"""
    teacher.eval()
    student.eval()
    
    # Get penultimate layer features
    teacher_feats = []
    student_feats = []
    labels = []
    
    # Hook to get features before FC
    def get_features(model, layer_name='avgpool'):
        features = []
        def hook(module, input, output):
            features.append(output.detach().flatten(1))
        
        modules = dict(model.named_modules())
        handle = modules[layer_name].register_forward_hook(hook)
        return features, handle
    
    t_feats, t_handle = get_features(teacher)
    s_feats, s_handle = get_features(student)
    
    with torch.no_grad():
        for images, lbls in loader:
            images = images.to(device)
            _ = teacher(images)
            _ = student(images)
            labels.extend([CIFAR_CLASSES[l] for l in lbls.numpy()])
            
            if len(labels) >= 500:  # Limit for visualization
                break
    
    t_handle.remove()
    s_handle.remove()
    
    teacher_features = torch.cat(t_feats, dim=0)[:500]
    student_features = torch.cat(s_feats, dim=0)[:500]
    labels = labels[:500]
    
    # Log to TensorBoard
    tb_logger.log_embedding(teacher_features, labels, f"teacher_epoch_{epoch}", epoch)
    tb_logger.log_embedding(student_features, labels, f"student_epoch_{epoch}", epoch)


# ============================================================================
# CHECKPOINTING
# ============================================================================

def save_checkpoint(model: nn.Module, path: str, extra_info: dict = None):
    """Save model checkpoint"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'timestamp': time.time()
    }
    
    if extra_info:
        checkpoint.update(extra_info)
    
    torch.save(checkpoint, path)
    print(f"💾 Saved checkpoint: {path}")


def load_checkpoint(model: nn.Module, path: str, device: torch.device = None) -> dict:
    """Load model checkpoint"""
    if device is None:
        device = get_device()
    
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"📂 Loaded checkpoint: {path}")
    
    return checkpoint


# ============================================================================
# RESULTS LOGGING
# ============================================================================

class ExperimentLogger:
    """Log experiment results to JSON"""
    
    def __init__(self, path: str = "reports/kd_results.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records = []
    
    def log(self, **kwargs):
        record = {'timestamp': time.time(), **kwargs}
        self.records.append(record)
    
    def save(self):
        with open(self.path, 'w') as f:
            json.dump(self.records, f, indent=2)
        print(f"📝 Saved results: {self.path}")
    
    def load(self):
        if self.path.exists():
            with open(self.path, 'r') as f:
                self.records = json.load(f)




# ============================================================================
# FEATURE VISUALIZATION (TRAJECTORIES)
# ============================================================================

def extract_features(model: nn.Module, loader: DataLoader, device: torch.device, 
                    layer_name: str = 'avgpool', max_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Extract intermediate features from a model"""
    model.eval()
    features_list = []
    labels_list = []
    
    # helper for hooking
    curr_features = []
    def hook_fn(module, input, output):
        curr_features.append(output.detach().flatten(1).cpu())
        
    # Register hook
    handle = None
    for name, module in model.named_modules():
        if name == layer_name:
            handle = module.register_forward_hook(hook_fn)
            break
            
    if handle is None:
        # Fallback for models where we can't find the layer easily, or explicit 'fc'
        print(f"⚠️ Layer {layer_name} not found. Returning None.")
        return None, None

    count = 0
    with torch.no_grad():
        for images, labels in loader:
            curr_features = [] # reset
            images = images.to(device)
            _ = model(images)
            
            if curr_features:
                features_list.append(curr_features[0])
                labels_list.append(labels)
                count += len(labels)
            
            if count >= max_samples:
                break
                
    handle.remove()
    
    if not features_list:
        return np.array([]), np.array([])
        
    features = torch.cat(features_list, dim=0)[:max_samples].numpy()
    labels = torch.cat(labels_list, dim=0)[:max_samples].numpy()
    
    return features, labels

def plot_feature_comparison(teacher: nn.Module, student: nn.Module, 
                           loader: DataLoader, device: torch.device,
                           method: str = 'tsne', dims: int = 2,
                           max_samples: int = 1000,
                           title: str = "Feature Space Comparison"):
    """
    Visualize Teacher vs Student feature space.
    method: 'tsne' or 'pca'
    dims: 2 or 3
    """
    print(f"Extracting features (max {max_samples} samples)...")
    
    # Try different layer names if defaults fail
    t_layer = 'avgpool' if hasattr(teacher, 'avgpool') else 'layer4'
    s_layer = 'avgpool' if hasattr(student, 'avgpool') else 'layer4'
    
    t_feats, t_labels = extract_features(teacher, loader, device, t_layer, max_samples)
    s_feats, s_labels = extract_features(student, loader, device, s_layer, max_samples)
    
    if len(t_feats) == 0 or len(s_feats) == 0:
        print("Error extracting features.")
        return

    # Project separately since dimensions may differ (ResNet50: 2048, ResNet18: 512)
    print(f"Computing {method.upper()} projection...")
    if method == 'tsne':
        projector = TSNE(n_components=dims, random_state=42, init='pca', learning_rate='auto')
    else:
        projector = PCA(n_components=dims, random_state=42)
    
    # Fit on combined data, but handle dimension mismatch
    # Strategy: Project to common PCA space first if dimensions differ
    if t_feats.shape[1] != s_feats.shape[1]:
        print(f"Dimension mismatch detected: Teacher={t_feats.shape[1]}, Student={s_feats.shape[1]}")
        print("Projecting to common PCA space first...")
        
        # Use PCA to reduce both to same dimension (min of the two)
        common_dim = min(t_feats.shape[1], s_feats.shape[1], 128)  # Cap at 128 for efficiency
        pca_teacher = PCA(n_components=common_dim, random_state=42)
        pca_student = PCA(n_components=common_dim, random_state=42)
        
        t_feats = pca_teacher.fit_transform(t_feats)
        s_feats = pca_student.fit_transform(s_feats)
    
    # Now concatenate and project
    n_t = len(t_feats)
    combined_feats = np.concatenate([t_feats, s_feats], axis=0)
    projections = projector.fit_transform(combined_feats)
    
    # Prepare DataFrame
    df = pd.DataFrame(projections, columns=[f'Dim{i+1}' for i in range(dims)])
    df['Label'] = np.concatenate([t_labels, s_labels])
    df['LabelName'] = [CIFAR_CLASSES[l] for l in df['Label']]
    df['Model'] = ['Teacher'] * n_t + ['Student'] * len(s_feats)
    
    # Plotting
    if dims == 2:
        plt.close('all')
        setup_solarized_style()
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        for i, model_name in enumerate(['Teacher', 'Student']):
            data = df[df['Model'] == model_name]
            sns.scatterplot(data=data, x='Dim1', y='Dim2', hue='LabelName', 
                          palette='tab10', ax=axes[i], legend=(i==1), s=60, alpha=0.7)
            axes[i].set_title(f"{model_name} ({method.upper()})")
            axes[i].grid(True, alpha=0.3)
            
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()
        
    elif dims == 3:
        # Interactive 3D plot with Plotly
        fig = px.scatter_3d(df, x='Dim1', y='Dim2', z='Dim3',
                           color='LabelName', symbol='Model',
                           title=f"{title} (3D {method.upper()})",
                           opacity=0.7, size_max=5)
        
        # Use simple IFrame display for Colab
        import IPython
        filename = "features_3d.html"
        fig.write_html(filename)
        display(IPython.display.IFrame(filename, width="100%", height=700))

def plot_all_models_comparison(teacher, students_dict, loader, device, 
                               method='tsne', dims=2, max_samples=1000,
                               title="All Models Comparison"):
    """
    Visualize Teacher + multiple Students on the same plot.
    students_dict: dict like {'Baseline': model1, 'KD Scores': model2, ...}
    """
    print(f"Extracting features from all models (max {max_samples} samples)...")
    
    all_features = []
    all_labels = []
    all_model_names = []
    
    # Extract Teacher features
    t_layer = 'avgpool' if hasattr(teacher, 'avgpool') else 'layer4'
    t_feats, t_labels = extract_features(teacher, loader, device, t_layer, max_samples)
    
    if len(t_feats) == 0:
        print("Error extracting Teacher features.")
        return
    
    all_features.append(t_feats)
    all_labels.append(t_labels)
    all_model_names.extend(['Teacher'] * len(t_feats))
    
    # Extract all Students features
    for student_name, student_model in students_dict.items():
        s_layer = 'avgpool' if hasattr(student_model, 'avgpool') else 'layer4'
        s_feats, s_labels = extract_features(student_model, loader, device, s_layer, max_samples)
        
        if len(s_feats) == 0:
            print(f"Error extracting {student_name} features.")
            continue
            
        all_features.append(s_feats)
        all_labels.append(s_labels)
        all_model_names.extend([student_name] * len(s_feats))
    
    # Handle dimension mismatches with PCA
    max_dim = max(f.shape[1] for f in all_features)
    min_dim = min(f.shape[1] for f in all_features)
    
    if max_dim != min_dim:
        print(f"Dimension mismatch detected (range: {min_dim}-{max_dim})")
        print("Projecting to common PCA space first...")
        
        common_dim = min(min_dim, 128)
        aligned_features = []
        for feats in all_features:
            pca = PCA(n_components=common_dim, random_state=42)
            aligned_features.append(pca.fit_transform(feats))
        all_features = aligned_features
    
    # Concatenate all
    combined_feats = np.concatenate(all_features, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)
    
    # Project
    print(f"Computing {method.upper()} projection...")
    if method == 'tsne':
        projector = TSNE(n_components=dims, random_state=42, init='pca', learning_rate='auto')
    else:
        projector = PCA(n_components=dims, random_state=42)
        
    projections = projector.fit_transform(combined_feats)
    
    # Create DataFrame
    df = pd.DataFrame(projections, columns=[f'Dim{i+1}' for i in range(dims)])
    df['Label'] = combined_labels
    df['LabelName'] = [CIFAR_CLASSES[l] for l in df['Label']]
    df['Model'] = all_model_names
    
    # Plot
    plt.close('all')
    setup_solarized_style()
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Plot each model with different marker
    markers = {'Teacher': 'o', 'Baseline': 's', 'KD Scores': '^', 'KD Features': 'D'}
    
    for model_name in df['Model'].unique():
        data = df[df['Model'] == model_name]
        marker = markers.get(model_name, 'x')
        sns.scatterplot(data=data, x='Dim1', y='Dim2', hue='LabelName', 
                       palette='tab10', ax=ax, 
                       marker=marker, s=80, alpha=0.6, legend=False)  # Removed label parameter
    
    ax.set_title(f"{title} ({method.upper()})", fontsize=16)
    ax.grid(True, alpha=0.3)
    
    # Custom legend (model types)
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker=markers.get(m, 'x'), color='gray', 
                             label=m, markersize=10, linestyle='None') 
                      for m in df['Model'].unique()]
    ax.legend(handles=legend_elements, title='Model', loc='upper right')
    
    plt.tight_layout()
    plt.show()