# %% 1. Importation des bibliothèques
import os
import json
import time
import random
import hashlib
import platform
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm.auto import tqdm

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign, box_iou
from torch.utils.data import Dataset
from matplotlib.patches import Rectangle

NAME = "Faster_RCNN"
EPOCHS = 3
BATCH_SIZE = 16
LR = 0.001
# %% 2. Configuration du GPU et des paramètres
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0), flush=True)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    print("ATTENTION : CUDA indisponible ; exécution sur CPU, vérifier le pilote et les permissions.", flush=True)
AMP = device.type == "cuda"
# Le chemin fonctionne aussi dans une fenêtre interactive VS Code.
BASE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if BASE.name != "TP_PyTorch" and (BASE / "TP_PyTorch").is_dir():
    BASE = BASE / "TP_PyTorch"
os.environ.setdefault("TORCH_HOME", str(BASE / "data" / "torch"))
FIGURES = BASE / "figures" / NAME
FIGURES.mkdir(parents=True, exist_ok=True)
(BASE / "metrics").mkdir(exist_ok=True)

def afficher(nom):
    plt.tight_layout()
    plt.savefig(FIGURES / f"{nom}.png", dpi=140)
    plt.show(block=False)
    plt.pause(0.001)
    plt.close()

# %% 3. Chargement du dataset MNIST
# Le seuil > 0 englobe tous les pixels du chiffre ; xmax/ymax sont exclusifs.
class MNISTDetection(Dataset):
    def __init__(self, train):
        self.mnist = datasets.MNIST(BASE / "data", train=train, download=True)

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, index):
        image, digit = self.mnist[index]
        image = transforms.functional.to_tensor(image)
        yy, xx = torch.where(image[0] > 0)
        if len(xx) == 0:
            raise ValueError(f"Image MNIST sans premier plan : {index}")
        box = torch.tensor([[xx.min(), yy.min(), xx.max() + 1, yy.max() + 1]], dtype=torch.float32)
        target = {"boxes": box, "labels": torch.tensor([digit + 1], dtype=torch.int64),
                  "image_id": torch.tensor(index), "area": (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1]),
                  "iscrowd": torch.zeros(1, dtype=torch.int64)}
        # Une copie contiguë permet le transfert en mémoire épinglée.
        return image.expand(3, -1, -1).contiguous(), target

full_train = MNISTDetection(True)
train_set, val_set = random_split(full_train, [50000, 10000], generator=torch.Generator().manual_seed(SEED))
test_set = MNISTDetection(False)
split_hash = hashlib.sha256(np.asarray(train_set.indices, dtype=np.int64).tobytes()).hexdigest()

def assembler(batch):
    return tuple(zip(*batch))

options = dict(batch_size=BATCH_SIZE, num_workers=0, pin_memory=AMP, collate_fn=assembler)
train_loader = DataLoader(train_set, shuffle=True, generator=torch.Generator().manual_seed(SEED), **options)
val_loader = DataLoader(val_set, shuffle=False, **options)
test_loader = DataLoader(test_set, shuffle=False, **options)
print(f"Effectifs : 50000 / 10000 / 10000 ; empreinte : {split_hash}", flush=True)

# %% 4. Exploration et visualisation des données
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for ax, index in zip(axes.flat, train_set.indices[:10]):
    image, target = full_train[index]
    ax.imshow(image[0], cmap="gray")
    x1, y1, x2, y2 = target["boxes"][0].tolist()
    ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor="lime"))
    ax.set_title(f"Chiffre {target['labels'].item()-1}")
    ax.axis("off")
afficher("donnees")

# %% 5. Définition de l'architecture du modèle
# Petit backbone appris de zéro : pas 4 au total, 64 canaux.
backbone = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
backbone.out_channels = 64
# RPN : ancres de plusieurs tailles et rapports, puis régression des propositions.
anchors = AnchorGenerator(sizes=((8, 16, 24, 32),), aspect_ratios=((0.5, 1.0, 2.0),))
# ROI Align extrait une grille 3×3 par proposition ; tête classe + boîte.
pool = MultiScaleRoIAlign(featmap_names=["0"], output_size=3, sampling_ratio=2)
model = FasterRCNN(backbone, num_classes=None, min_size=28, max_size=28,
    size_divisible=4, image_mean=[0.1307]*3, image_std=[0.3081]*3,
    rpn_anchor_generator=anchors, box_roi_pool=pool,
    box_head=TwoMLPHead(64*3*3, 128), box_predictor=FastRCNNPredictor(128, 11),
    rpn_pre_nms_top_n_train=200, rpn_post_nms_top_n_train=100,
    rpn_pre_nms_top_n_test=100, rpn_post_nms_top_n_test=50,
    rpn_batch_size_per_image=64, box_batch_size_per_image=64,
    box_detections_per_img=10, box_score_thresh=0.05).to(device)
print(model, flush=True)
trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_parameters = sum(p.numel() for p in model.parameters())
print(f"Paramètres entraînables : {trainable_parameters}", flush=True)

# %% 6. Configuration de l'entraînement
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scaler = torch.amp.GradScaler("cuda", enabled=AMP)
history = {k: [] for k in ["train_loss", "val_loss", "val_accuracy"]}

def deplacer(images, targets):
    return [x.to(device) for x in images], [{k: v.to(device) for k, v in t.items()} for t in targets]

def evaluer_detection(loader):
    model.eval()
    labels, predictions, ious, localized = [], [], [], []
    missing, examples = 0, []
    with torch.inference_mode():
        for images, targets in loader:
            images, targets = deplacer(images, targets)
            # Les opérateurs de boîtes restent en float32 pour la stabilité.
            outputs = model(images)
            for target, output in zip(targets, outputs):
                truth = target["labels"].item() - 1
                labels.append(truth)
                if len(output["scores"]) == 0:
                    predictions.append(-1)
                    missing += 1
                    ious.append(0.0)
                    localized.append(False)
                    if len(examples) < 12:
                        examples.append(None)
                else:
                    best = output["scores"].argmax()
                    prediction = output["labels"][best].item() - 1
                    predictions.append(prediction)
                    iou = box_iou(output["boxes"][best:best+1], target["boxes"])[0, 0].item()
                    ious.append(iou)
                    localized.append(prediction == truth and iou >= 0.5)
                    if len(examples) < 12:
                        examples.append((output["boxes"][best].cpu().tolist(), output["scores"][best].item()))
    return labels, predictions, missing, float(np.mean(ious)), float(np.mean(localized)), examples

def perte_validation():
    # Torchvision ne retourne les pertes qu'en mode train. Sans BatchNorm/dropout ici.
    # Le tirage des propositions est isolé : il ne modifie pas l'aléatoire d'entraînement.
    total = 0.0
    with torch.random.fork_rng(devices=[0] if AMP else []):
        torch.manual_seed(SEED)
        model.train()
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = deplacer(images, targets)
                with torch.autocast(device_type=device.type, enabled=AMP):
                    losses = model(images, targets)
                total += sum(losses.values()).item() * len(images)
    model.eval()
    return total / len(val_set)

# %% 7. Entraînement du modèle
if AMP:
    torch.cuda.synchronize()
start = time.perf_counter()
for epoch in range(EPOCHS):
    model.train()
    total = 0.0
    for images, targets in tqdm(train_loader, desc=f"Époque {epoch+1}/{EPOCHS}", mininterval=20):
        images, targets = deplacer(images, targets)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=AMP):
            losses = model(images, targets)
            loss = sum(losses.values())
        if not torch.isfinite(loss):
            raise RuntimeError("Perte de détection non finie")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * len(images)
    history["train_loss"].append(total / len(train_set))
    history["val_loss"].append(perte_validation())
    labels, predictions, *_ = evaluer_detection(val_loader)
    history["val_accuracy"].append(accuracy_score(labels, predictions))
    print(f"Époque {epoch+1} : " + json.dumps({k:v[-1] for k,v in history.items()}), flush=True)
if AMP:
    torch.cuda.synchronize()
training_seconds = time.perf_counter() - start

# %% 8. Visualisation des courbes d'apprentissage
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(range(1, EPOCHS+1), history["train_loss"], label="Entraînement")
axes[0].plot(range(1, EPOCHS+1), history["val_loss"], label="Validation")
axes[0].set_title("Somme des quatre pertes de détection")
axes[0].legend()
axes[1].plot(range(1, EPOCHS+1), history["val_accuracy"], label="Validation")
axes[1].set_title("Exactitude image par image")
axes[1].legend()
for ax in axes:
    ax.set_xlabel("Époque")
afficher("courbes")

# %% 9. Évaluation sur le jeu de test
y_true, y_pred, missing, mean_iou, localization_accuracy, examples = evaluer_detection(test_loader)
# Les absences (-1) sont des faux négatifs pour la classe réelle, jamais ignorées.
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(10)), average="macro", zero_division=0)
metrics = dict(model=NAME, status="complete", seed=SEED, split_sha256=split_hash,
    train_samples=50000, validation_samples=10000, test_samples=10000,
    epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LR,
    accuracy=accuracy_score(y_true,y_pred), macro_precision=float(precision),
    macro_recall=float(recall), macro_f1=float(f1), no_detection=missing,
    mean_top_detection_iou=mean_iou, correct_class_and_iou50=localization_accuracy,
    final_train_loss=history["train_loss"][-1], final_validation_loss=history["val_loss"][-1],
    training_seconds=training_seconds, timing_includes_validation=True,
    trainable_parameters=trainable_parameters, total_parameters=total_parameters,
    device=str(device), gpu=torch.cuda.get_device_name(0) if AMP else None,
    python=platform.python_version(), torch=torch.__version__,
    torchvision=__import__("torchvision").__version__, cuda=torch.version.cuda, amp=AMP, peak_gpu_memory_mb=torch.cuda.max_memory_allocated()/1024**2 if AMP else None, history=history)
print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)

# %% 10. Matrice de confusion et prédictions
# Une onzième colonne rend explicites les absences de détection.
cm = confusion_matrix(y_true, y_pred, labels=list(range(10))+[-1])[:10, :]
plt.figure(figsize=(9, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=list(range(10))+["Absence"], yticklabels=range(10))
plt.xlabel("Prédiction")
plt.ylabel("Vérité")
afficher("confusion")
fig, axes = plt.subplots(3, 4, figsize=(11, 8))
for index, ax in enumerate(axes.flat):
    image, target = test_set[index]
    ax.imshow(image[0], cmap="gray")
    if examples[index] is not None:
        (x1,y1,x2,y2), confidence = examples[index]
        ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor="lime"))
        ax.set_title(f"Vrai {y_true[index]}, prédit {y_pred[index]} ({confidence:.2f})")
    else:
        ax.set_title(f"Vrai {y_true[index]} : aucune détection")
    ax.axis("off")
afficher("predictions")

# %% 11. Sauvegarde des résultats
metrics["confusion_matrix"] = cm.tolist()
with (BASE / "metrics" / f"{NAME}.json").open("w") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False, allow_nan=False)
print("Expérience terminée : Faster R-CNN", flush=True)
