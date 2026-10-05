"""CPU CNN baseline and frozen MobileNet transfer architecture."""

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.transforms import v2


def transform(size: int = 160):
    return v2.Compose(
        [
            v2.Resize((size, size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 12, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(12, 24, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1),
            nn.ReLU(),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(48, 3))

    def forward(self, x):
        return self.classifier(self.avgpool(self.features(x)).flatten(1))


def build(kind: str, pretrained: bool = False):
    if kind == "small_cnn":
        return SmallCNN()
    if kind != "mobilenet_transfer":
        raise ValueError("Unknown model architecture")
    model = mobilenet_v3_small(
        weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    )
    for parameter in model.features.parameters():
        parameter.requires_grad_(False)
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(576, 3))
    return model


def gradcam(model, tensor, target: int):
    captured = []

    def hook(_module, _args, output):
        output.retain_grad()
        captured.append(output)

    handle = model.features.register_forward_hook(hook)
    model.eval()
    model.zero_grad(set_to_none=True)
    tensor = tensor.detach().requires_grad_(True)
    try:
        scores = model(tensor)
        scores[0, target].backward()
        activation = captured[0]
        weights = activation.grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activation).sum(dim=1).relu()[0].detach()
        cam = cam / cam.max().clamp_min(1e-8)
        return cam.numpy()
    finally:
        handle.remove()
