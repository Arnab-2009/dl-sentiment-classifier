"""
Deep Learning Customer Review Sentiment Classifier (PyTorch)
-----------------------------------------------------------
Builds a custom vocabulary tokenizer, trains a PyTorch Neural Network
(Embedding -> Global Average Pooling -> Linear -> ReLU -> Dropout -> Output),
evaluates test accuracy, and exports training curves and confusion matrix plots.
"""

import os
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_review_dataset(n_samples: int = 1800, seed: int = 42) -> tuple[list[str], list[int]]:
    """Generate a realistic synthetic dataset of positive (1) and negative (0) product reviews."""
    rng = np.random.default_rng(seed)

    pos_templates = [
        "absolutely love this {item} the build quality is {adj_pos} and delivery was {speed_pos}",
        "{adj_pos} performance works right out of the box highly recommend this {item}",
        "best {item} I have bought this year super {adj_pos} and worth every rupee",
        "customer support was {adj_pos} and the {item} exceeded my expectations",
        "smooth setup {adj_pos} battery life and {speed_pos} shipping very satisfied",
        "five stars for this {item} truly {adj_pos} design and reliable daily use",
    ]
    neg_templates = [
        "terrible {item} stopped working after two days completely {adj_neg} quality",
        "waste of money the {item} is {adj_neg} and delivery was {speed_neg}",
        "very disappointed with this {item} feels cheap and {adj_neg}",
        "awful experience customer support was {adj_neg} and refused a replacement",
        "do not buy this {item} overheats quickly and performs {adj_neg}",
        "one star broken on arrival and the {item} is totally {adj_neg}",
    ]

    items = ["headset", "keyboard", "monitor", "laptop stand", "router", "smartwatch", "charger", "mouse"]
    adj_pos = ["amazing", "fantastic", "solid", "reliable", "fast", "impressive", "clean", "durable"]
    adj_neg = ["broken", "useless", "slow", "defective", "fragile", "horrible", "laggy", "poor"]
    speed_pos = ["super fast", "on time", "quick", "smooth"]
    speed_neg = ["delayed", "late", "painfully slow", "terrible"]

    texts, labels = [], []
    for _ in range(n_samples):
        is_positive = int(rng.random() > 0.5)
        item = rng.choice(items)
        if is_positive:
            template = rng.choice(pos_templates)
            text = template.format(
                item=item,
                adj_pos=rng.choice(adj_pos),
                speed_pos=rng.choice(speed_pos),
            )
        else:
            template = rng.choice(neg_templates)
            text = template.format(
                item=item,
                adj_neg=rng.choice(adj_neg),
                speed_neg=rng.choice(speed_neg),
            )
        texts.append(text)
        labels.append(is_positive)

    return texts, labels


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def build_vocab(texts: list[str], max_vocab: int = 1000) -> dict[str, int]:
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counter.most_common(max_vocab - 2):
        vocab[word] = len(vocab)
    return vocab


def encode_texts(texts: list[str], vocab: dict[str, int], max_len: int = 20) -> np.ndarray:
    encoded = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, text in enumerate(texts):
        tokens = tokenize(text)[:max_len]
        for j, tok in enumerate(tokens):
            encoded[i, j] = vocab.get(tok, vocab["<UNK>"])
    return encoded


class SentimentNet(nn.Module):
    """Lightweight Neural Network for Text Classification."""

    def __init__(self, vocab_size: int, embed_dim: int = 32, hidden_dim: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        pooled = embedded.mean(dim=1)  # Global average pooling across tokens
        hidden = self.dropout(self.relu(self.fc1(pooled)))
        return self.fc2(hidden).squeeze(1)


def save_plots(
    history: dict[str, list[float]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str = "outputs",
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Training & Validation Loss / Accuracy Curves
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", linewidth=2, color="#2563EB")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", linewidth=2, color="#DC2626")
    axes[0].set_title("Binary Cross-Entropy Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train Acc", linewidth=2, color="#2563EB")
    axes[1].plot(epochs, history["val_acc"], label="Val Acc", linewidth=2, color="#16A34A")
    axes[1].set_title("Classification Accuracy", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=300)
    plt.close()

    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Purples",
        xticklabels=["Negative (0)", "Positive (1)"],
        yticklabels=["Negative (0)", "Positive (1)"],
    )
    plt.title("PyTorch Sentiment Classifier — Confusion Matrix", fontweight="bold")
    plt.xlabel("Predicted Sentiment")
    plt.ylabel("Actual Sentiment")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
    plt.close()


def main() -> None:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"1. Preparing dataset on device: {device}")

    texts, labels = generate_review_dataset(n_samples=1800, seed=42)
    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    vocab = build_vocab(X_train_txt)
    X_train = torch.tensor(encode_texts(X_train_txt, vocab), dtype=torch.long)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(encode_texts(X_test_txt, vocab), dtype=torch.long)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, y_train_t), batch_size=32, shuffle=True)

    model = SentimentNet(vocab_size=len(vocab), embed_dim=32, hidden_dim=32).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print("2. Training PyTorch neural network...\n")
    for epoch in range(1, 16):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_y)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == batch_y).sum().item()
            total += len(batch_y)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_test.to(device))
            val_loss = criterion(val_logits, y_test_t.to(device)).item()
            val_preds = (torch.sigmoid(val_logits) >= 0.5).float().cpu().numpy()
            val_acc = float((val_preds == np.array(y_test)).mean())

        train_loss = total_loss / total
        train_acc = correct / total
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if epoch % 3 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:02d}/15 | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

    print("\n3. Final Test Evaluation:")
    print(classification_report(y_test, val_preds, target_names=["Negative", "Positive"]))

    save_plots(history, np.array(y_test), val_preds)
    print("4. Saved plots to ./outputs/ (training_curves.png & confusion_matrix.png)")

    # Quick live inference demo
    sample_reviews = [
        "amazing headset super fast shipping and solid sound",
        "terrible keyboard arrived broken and completely useless",
    ]
    sample_tensor = torch.tensor(encode_texts(sample_reviews, vocab), dtype=torch.long).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(sample_tensor)).cpu().numpy()
    print("\n5. Sample Inference:")
    for rev, prob in zip(sample_reviews, probs):
        sentiment = "Positive" if prob >= 0.5 else "Negative"
        print(f"   [{sentiment} | {prob:.2%}] -> \"{rev}\"")


if __name__ == "__main__":
    main()