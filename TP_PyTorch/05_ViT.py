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

NAME = "ViT"
EPOCHS = 8
BATCH_SIZE = 128
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

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# %% 3. Chargement du dataset MNIST
full_train = datasets.MNIST(BASE / "data", train=True, download=True, transform=transform)
train_set, val_set = random_split(full_train, [50000, 10000], generator=torch.Generator().manual_seed(SEED))
test_set = datasets.MNIST(BASE / "data", train=False, download=True, transform=transform)
split_hash = hashlib.sha256(np.asarray(train_set.indices, dtype=np.int64).tobytes()).hexdigest()
loader_options = dict(batch_size=BATCH_SIZE, num_workers=0, pin_memory=AMP)
train_loader = DataLoader(train_set, shuffle=True, generator=torch.Generator().manual_seed(SEED), **loader_options)
val_loader = DataLoader(val_set, shuffle=False, **loader_options)
test_loader = DataLoader(test_set, shuffle=False, **loader_options)
print(f"Effectifs : {len(train_set)}, {len(val_set)}, {len(test_set)} ; empreinte : {split_hash}", flush=True)

# %% 4. Exploration et visualisation des données
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for ax, index in zip(axes.flat, train_set.indices[:10]):
    ax.imshow(full_train.data[index], cmap="gray")
    ax.set_title(f"Chiffre {full_train.targets[index].item()}")
    ax.axis("off")
afficher("donnees")

# %% 5. Définition de l'architecture du modèle
# Chaque image fournit 16 patches 7×7 ; le jeton de classe porte la décision.
class Attention(nn.Module):
    def __init__(self, dim=64, heads=4):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.projection = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.heads, dim // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        # Softmax en précision simple pour stabiliser les scores d'attention.
        scores = (q.float() @ k.float().transpose(-2, -1)) / (dim // self.heads) ** 0.5
        weights = self.dropout(scores.softmax(dim=-1)).to(v.dtype)
        return self.projection((weights @ v).transpose(1, 2).reshape(batch, tokens, dim))

class BlocEncodeur(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(64)
        self.attention = Attention()
        self.norm2 = nn.LayerNorm(64)
        self.mlp = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(128, 64), nn.Dropout(0.1))

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))

class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embedding = nn.Linear(49, 64)
        self.cls = nn.Parameter(torch.zeros(1, 1, 64))
        self.position = nn.Parameter(torch.randn(1, 17, 64) * 0.02)
        self.blocks = nn.Sequential(*[BlocEncodeur() for _ in range(4)])
        self.norm = nn.LayerNorm(64)
        self.head = nn.Linear(64, 10)

    def forward(self, images):
        patches = images.unfold(2, 7, 7).unfold(3, 7, 7)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(images.size(0), 16, 49)
        tokens = self.patch_embedding(patches)
        tokens = torch.cat([self.cls.expand(images.size(0), -1, -1), tokens], dim=1)
        return self.head(self.norm(self.blocks(tokens + self.position))[:, 0])

model = VisionTransformer().to(device)
print(model, flush=True)
trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_parameters = sum(p.numel() for p in model.parameters())
print(f"Paramètres entraînables : {trainable_parameters:,} / {total_parameters:,}", flush=True)

# %% 6. Configuration de l'entraînement
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scaler = torch.amp.GradScaler("cuda", enabled=AMP)
history = {k: [] for k in ["train_loss", "val_loss", "train_accuracy", "val_accuracy"]}

def evaluer(loader):
    model.eval()
    total_loss, labels, predictions = 0.0, [], []
    with torch.inference_mode():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            with torch.autocast(device_type=device.type, enabled=AMP):
                logits = model(images)
                loss = criterion(logits, targets)
            total_loss += loss.item() * len(targets)
            labels.extend(targets.cpu().tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
    return total_loss / len(loader.dataset), labels, predictions

# %% 7. Entraînement du modèle
# Budget fixé avant l'évaluation du test ; aucune sélection sur le test.
if AMP:
    torch.cuda.synchronize()
start = time.perf_counter()
for epoch in range(EPOCHS):
    model.train()
    running_loss, correct = 0.0, 0
    for images, targets in tqdm(train_loader, desc=f"Époque {epoch + 1}/{EPOCHS}", mininterval=20):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=AMP):
            logits = model(images)
            loss = criterion(logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("Perte non finie pendant l'entraînement")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * len(targets)
        correct += (logits.argmax(1) == targets).sum().item()
    val_loss, val_labels, val_predictions = evaluer(val_loader)
    history["train_loss"].append(running_loss / len(train_set))
    history["val_loss"].append(val_loss)
    history["train_accuracy"].append(correct / len(train_set))
    history["val_accuracy"].append(accuracy_score(val_labels, val_predictions))
    print(f"Époque {epoch + 1} : " + json.dumps({k: v[-1] for k, v in history.items()}), flush=True)
if AMP:
    torch.cuda.synchronize()
training_seconds = time.perf_counter() - start

# %% 8. Visualisation des courbes d'apprentissage
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for split, label in [("train", "Entraînement"), ("val", "Validation")]:
    axes[0].plot(range(1, EPOCHS + 1), history[f"{split}_loss"], label=label)
    axes[1].plot(range(1, EPOCHS + 1), history[f"{split}_accuracy"], label=label)
for ax, title in zip(axes, ["Entropie croisée", "Exactitude"]):
    ax.set(xlabel="Époque", title=title)
    ax.legend()
afficher("courbes")

# %% 9. Évaluation sur le jeu de test
test_loss, y_true, y_pred = evaluer(test_loader)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(10)), average="macro", zero_division=0)
metrics = dict(model=NAME, status="complete", seed=SEED, split_sha256=split_hash,
               train_samples=len(train_set), validation_samples=len(val_set), test_samples=len(test_set),
               epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LR,
               accuracy=accuracy_score(y_true, y_pred), macro_precision=float(precision),
               macro_recall=float(recall), macro_f1=float(f1), test_loss=test_loss,
               final_train_loss=history["train_loss"][-1], final_validation_loss=history["val_loss"][-1],
               training_seconds=training_seconds, timing_includes_validation=True,
               trainable_parameters=trainable_parameters, total_parameters=total_parameters,
               device=str(device), gpu=torch.cuda.get_device_name(0) if AMP else None,
               python=platform.python_version(), torch=torch.__version__,
               torchvision=__import__("torchvision").__version__, cuda=torch.version.cuda,
               amp=AMP, peak_gpu_memory_mb=torch.cuda.max_memory_allocated()/1024**2 if AMP else None, history=history)
print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)

# %% 10. Matrice de confusion et prédictions
cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.xlabel("Prédiction")
plt.ylabel("Vérité")
plt.title(NAME)
afficher("confusion")
fig, axes = plt.subplots(3, 4, figsize=(10, 7))
for index, ax in enumerate(axes.flat):
    ax.imshow(test_set.data[index], cmap="gray")
    ax.set_title(f"Vrai : {y_true[index]} ; prédit : {y_pred[index]}")
    ax.axis("off")
afficher("predictions")

# %% 11. Sauvegarde des résultats
metrics["confusion_matrix"] = cm.tolist()
with (BASE / "metrics" / f"{NAME}.json").open("w") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False, allow_nan=False)
print(f"Expérience terminée : {NAME} ; résultats dans {BASE / 'metrics'}", flush=True)
