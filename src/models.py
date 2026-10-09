"""
Модели:
  * BaselineCNN — простая свёрточная сеть «с нуля» (точка сравнения);
  * build_main_model — ResNet-18/50, предобученная на ImageNet, дообучение (fine-tuning).

Где взять веса ImageNet:
  1) стандартно — torchvision (download.pytorch.org);
  2) если сайт недоступен — зеркало весов timm на GitHub Releases
     (ResNet strikes back, Wightman et al. 2021). Имена слоёв timm-ResNet совпадают с torchvision.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

TIMM_GITHUB = {
    "resnet18": "https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet18_a1_0-d63eafa0.pth",
    "resnet50": "https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1_0-14fe96d1.pth",
}


class BaselineCNN(nn.Module):
    """4 блока Conv-BN-ReLU-MaxPool + Global Average Pooling + Dropout + Linear. ~0.4 млн параметров."""

    def __init__(self, num_classes: int, width: int = 32):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        w = width
        self.features = nn.Sequential(block(3, w), block(w, 2 * w), block(2 * w, 4 * w), block(4 * w, 8 * w))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(8 * w, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


def _load_imagenet_weights(arch: str) -> dict:
    try:
        w = {"resnet18": models.ResNet18_Weights.IMAGENET1K_V1, "resnet50": models.ResNet50_Weights.IMAGENET1K_V2}[arch]
        sd = w.get_state_dict(progress=True)
        print(f"Веса {arch}: torchvision {w}")
        return sd
    except Exception as e:  # сеть закрыта / нет интернета к download.pytorch.org
        print(f"torchvision-веса недоступны ({type(e).__name__}); беру зеркало timm с GitHub")
        sd = torch.hub.load_state_dict_from_url(TIMM_GITHUB[arch], progress=True, map_location="cpu")
        print(f"Веса {arch}: {TIMM_GITHUB[arch].rsplit('/', 1)[-1]}")
        return sd


def build_main_model(arch: str, num_classes: int, pretrained: bool = True, freeze_stem: bool = False) -> nn.Module:
    ctor = {"resnet18": models.resnet18, "resnet50": models.resnet50}[arch]
    m = ctor(weights=None)
    if pretrained:
        missing, unexpected = m.load_state_dict(_load_imagenet_weights(arch), strict=False)
        assert not [k for k in missing if not k.startswith("fc.")], missing
    m.fc = nn.Linear(m.fc.in_features, num_classes)  # новая «голова» под 37 пород
    if freeze_stem:
        # Нижние слои извлекают универсальные края/текстуры — их можно не дообучать.
        for mod in (m.conv1, m.bn1, m.layer1):
            for p in mod.parameters():
                p.requires_grad = False
    return m


def gradcam_target_layer(model: nn.Module):
    """Последний свёрточный блок — стандартный выбор слоя для Grad-CAM."""
    if isinstance(model, BaselineCNN):
        return model.features[-1]
    return model.layer4[-1]


def build_model(kind: str, num_classes: int, cfg: dict | None = None, pretrained: bool = True) -> nn.Module:
    if kind == "baseline_cnn":
        width = cfg["baseline_cnn"].get("width", 32) if cfg else 32
        return BaselineCNN(num_classes, width)
    mc = cfg["main_model"] if cfg else {"arch": "resnet18", "freeze_stem": False}
    return build_main_model(mc["arch"], num_classes, pretrained=pretrained, freeze_stem=mc.get("freeze_stem", False))


def load_checkpoint(path, device="cpu") -> tuple[nn.Module, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ck["kind"], ck["num_classes"], ck.get("cfg"), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, ck
