import torch
import torch.nn as nn
from tqdm import tqdm


def train_model(model, train_loader, device, epochs=10, lr=1e-3):
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0.0
        correct    = 0
        total      = 0

        # ── tqdm progress bar per epoch ──────────────────────────────
        loop = tqdm(
            train_loader,
            desc=f"Epoch [{epoch+1}/{epochs}]",
            leave=True,          # keeps bar after epoch finishes
            ncols=100            # bar width
        )

        for batch in loop:
            labels   = batch["label"].to(device)
            batch_in = {k: v.to(device) for k, v in batch.items()
                        if k != "label"}

            optimizer.zero_grad()
            outputs = model(batch_in)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

            # ── Live stats shown on the bar ──────────────────────────
            loop.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc" : f"{100.0 * correct / total:.2f}%"
            })

        epoch_loss = total_loss / len(train_loader)
        epoch_acc  = 100.0 * correct / total
        print(f"[Train] Epoch {epoch+1}/{epochs} "
              f"| Avg Loss: {epoch_loss:.4f} "
              f"| Acc: {epoch_acc:.2f}%")


def evaluate(model, loader, device):
    model.to(device)
    model.eval()

    correct = 0
    total   = 0

    loop = tqdm(loader, desc="Evaluating", leave=True, ncols=100)

    with torch.no_grad():
        for batch in loop:
            labels   = batch["label"].to(device)
            batch_in = {k: v.to(device) for k, v in batch.items()
                        if k != "label"}

            outputs = model(batch_in)
            preds   = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

            loop.set_postfix({
                "acc": f"{100.0 * correct / total:.2f}%"
            })

    acc = 100.0 * correct / total
    print(f"[Eval] Accuracy: {acc:.2f}%")
    return acc


def train_and_eval(model, train_loader, test_loader, device, epochs=10):
    train_model(model, train_loader, device, epochs=epochs)
    return evaluate(model, test_loader, device)