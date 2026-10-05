"""Validation-selected training, probability calibration and untouched test reporting."""

import copy
import hashlib
import json
import logging
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    log_loss,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from plantvision.data import CLASSES, audit, download
from plantvision.models import build, gradcam, transform


def probabilities(logits, temperature=1.0):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive and finite")
    return softmax(np.asarray(logits) / temperature, axis=1)


def metrics(labels, logits, temperature=1.0):
    p = probabilities(logits, temperature)
    confidence = p.max(axis=1)
    correct = p.argmax(axis=1) == labels
    ece = 0.0
    for low, high in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:], strict=True):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return {
        "accuracy": float(accuracy_score(labels, p.argmax(1))),
        "macro_f1": float(f1_score(labels, p.argmax(1), average="macro")),
        "log_loss": float(log_loss(labels, p, labels=[0, 1, 2])),
        "multiclass_brier": float(np.mean(np.sum((p - np.eye(3)[labels]) ** 2, axis=1))),
        "ece_10_bins": float(ece),
    }


def tensors(root, frame, size):
    tr = transform(size)
    values = []
    for path in frame.path:
        with Image.open(root / path) as image:
            values.append(tr(image.convert("RGB")))
    return torch.stack(values), torch.tensor(frame.label.to_numpy(), dtype=torch.long)


def logits(model, x, batch=32):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(chunk) for chunk in x.split(batch)]).numpy()


def fit(model, x, y, vx, vy, epochs, lr, augment=False):
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    best, history, best_score = None, [], -1.0
    loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True, num_workers=0)
    for epoch in range(epochs):
        model.train()
        for xx, yy in loader:
            if augment and torch.rand(()) < 0.5:
                xx = xx.flip(-1)
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(xx), yy)
            loss.backward()
            optimizer.step()
        score = metrics(vy.numpy(), logits(model, vx))["macro_f1"]
        history.append({"epoch": epoch + 1, "validation_macro_f1": score})
        if score > best_score:
            best, best_score = copy.deepcopy(model.state_dict()), score
        logging.info("Epoch %s/%s validation macro-F1 %.4f", epoch + 1, epochs, score)
    model.load_state_dict(best)
    return history


def run(root: Path):
    config = json.loads((root / "configs/default.json").read_text())
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    os.environ["TORCH_HOME"] = str(root / "models/pretrained")
    download(root)
    frame, evidence = audit(root)
    reports = root / "reports"
    (reports / "figures").mkdir(parents=True, exist_ok=True)
    frame.to_csv(reports / "image-manifest.csv", index=False)
    (reports / "data-audit.json").write_text(json.dumps(evidence, indent=2))
    split = {
        s: frame[frame.split == s].reset_index(drop=True) for s in ["train", "validation", "test"]
    }
    data = {s: tensors(root, f, config["image_size"]) for s, f in split.items()}
    x, y = data["train"]
    vx, vy = data["validation"]
    cnn = build("small_cnn")
    cnn_history = fit(cnn, x, y, vx, vy, config["cnn_epochs"], 0.001, augment=True)
    mobile = build("mobilenet_transfer", pretrained=True)
    mobile.eval()
    with torch.no_grad():

        def extract(xx):
            return torch.cat([mobile.avgpool(mobile.features(c)).flatten(1) for c in xx.split(32)])

        features = torch.cat([extract(x), extract(x.flip(-1))])
        validation_features = extract(vx)
    head_history = fit(
        mobile.classifier,
        features,
        torch.cat([y, y]),
        validation_features,
        vy,
        config["head_epochs"],
        0.005,
    )
    candidates = {"small_cnn": cnn, "mobilenet_transfer": mobile}
    validation = {
        name: metrics(vy.numpy(), logits(model, vx)) for name, model in candidates.items()
    }
    chosen = max(validation, key=lambda n: validation[n]["macro_f1"])
    selected = candidates[chosen]
    val_logits = logits(selected, vx)
    temperature = float(
        minimize_scalar(
            lambda t: log_loss(vy.numpy(), probabilities(val_logits, t), labels=[0, 1, 2]),
            bounds=(0.05, 5.0),
            method="bounded",
        ).x
    )
    tx, ty = data["test"]
    test_logits = {name: logits(model, tx) for name, model in candidates.items()}
    test = {name: metrics(ty.numpy(), z) for name, z in test_logits.items()}
    calibrated = metrics(ty.numpy(), test_logits[chosen], temperature)
    stress_x = []
    tr = transform(config["image_size"])
    for path in split["test"].path:
        with Image.open(root / path) as image:
            altered = ImageEnhance.Brightness(
                image.convert("RGB").filter(ImageFilter.GaussianBlur(2))
            ).enhance(0.7)
            stress_x.append(tr(altered))
    stress = metrics(ty.numpy(), logits(selected, torch.stack(stress_x)), temperature)
    model_path = root / "models/selected.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": selected.state_dict(),
            "kind": chosen,
            "temperature": temperature,
            "image_size": config["image_size"],
        },
        model_path,
    )
    output = {
        "config": config,
        "validation": validation,
        "selected_on_validation": chosen,
        "test_uncalibrated": test,
        "test_calibrated": calibrated,
        "test_blur_darkness_check": stress,
        "temperature_from_validation": temperature,
        "history": {"small_cnn": cnn_history, "mobilenet_transfer": head_history},
        "counts": evidence["retained_counts"],
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "versions": {"torch": torch.__version__, "numpy": np.__version__},
        "limitations": [
            "Supplied split has no plant/farm IDs; independence beyond exact hashes is unproven",
            "Altered test images are a robustness check, not external-domain validation",
            "Three bean classes only; softmax confidence does not detect other diseases or species",
        ],
    }
    (reports / "metrics.json").write_text(json.dumps(output, indent=2))
    p = probabilities(test_logits[chosen], temperature)
    predictions = split["test"][["path", "label"]].copy()
    for i, name in enumerate(CLASSES):
        predictions[name + "_probability"] = p[:, i]
    predictions["predicted_label"] = p.argmax(1)
    predictions.to_csv(reports / "test-predictions.csv", index=False)
    (reports / "classification-report.json").write_text(
        json.dumps(
            classification_report(ty.numpy(), p.argmax(1), target_names=CLASSES, output_dict=True),
            indent=2,
        )
    )
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    ConfusionMatrixDisplay.from_predictions(
        ty.numpy(),
        p.argmax(1),
        display_labels=CLASSES,
        ax=ax,
        cmap="Greens",
        colorbar=False,
        xticks_rotation=20,
    )
    ax.set_title("Untouched supplied test split · selected on validation")
    fig.savefig(reports / "figures/confusion-matrix.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(8, 10), layout="constrained")
    for label in range(3):
        index = int(np.flatnonzero(ty.numpy() == label)[0])
        with Image.open(root / split["test"].path.iloc[index]) as image:
            image = image.convert("RGB").resize((160, 160))
            axes[label, 0].imshow(image)
            axes[label, 0].set_title("True: " + CLASSES[label])
            cam = gradcam(selected, tx[index : index + 1], int(p[index].argmax()))
            heat = Image.fromarray((cam * 255).astype("uint8")).resize(
                (160, 160), Image.Resampling.BILINEAR
            )
            axes[label, 1].imshow(image)
            axes[label, 1].imshow(np.asarray(heat) / 255, cmap="jet", alpha=0.45, vmin=0, vmax=1)
            axes[label, 1].set_title(
                f"Predicted: {CLASSES[p[index].argmax()]} ({p[index].max():.2f})"
            )
        axes[label, 0].axis("off")
        axes[label, 1].axis("off")
    fig.suptitle("Grad-CAM · first test example per class · not causal evidence")
    fig.savefig(reports / "figures/gradcam.png", dpi=150)
    plt.close(fig)
    return output


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = run(Path.cwd())
    print(
        json.dumps(
            {"selected": result["selected_on_validation"], "test": result["test_calibrated"]},
            indent=2,
        )
    )
