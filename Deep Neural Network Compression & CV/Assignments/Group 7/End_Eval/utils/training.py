import torch
import torch.nn as nn
from tqdm import tqdm


def train_model(model, train_loader, device, epochs=10, lr=1e-3):
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = lr,
        weight_decay = 1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    for epoch in range(epochs):
        model.train()
        total_loss = correct = total = 0

        loop = tqdm(train_loader,
                    desc=f"Epoch [{epoch+1}/{epochs}]",
                    leave=True, ncols=100)

        for batch in loop:
            labels   = batch["label"].to(device)
            batch_in = {k: v.to(device) for k, v in batch.items()
                        if k != "label"}

            optimizer.zero_grad()
            outputs = model(batch_in)

            n_classes = outputs.size(1)
            if int(labels.min()) < 0 or int(labels.max()) >= n_classes:
                raise ValueError(
                    f"Label out of range: min={int(labels.min())}, "
                    f"max={int(labels.max())}, expected [0, {n_classes-1}]"
                )

            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

            loop.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc" : f"{100.0 * correct / total:.2f}%",
                "lr"  : f"{scheduler.get_last_lr()[0]:.6f}"
            })

        scheduler.step()
        print(f"[Train] Epoch {epoch+1}/{epochs} "
              f"| Loss: {total_loss/len(train_loader):.4f} "
              f"| Acc: {100.0*correct/total:.2f}% "
              f"| LR: {scheduler.get_last_lr()[0]:.6f}")


def evaluate(model, loader, device):
    model.to(device)
    model.eval()
    correct = total = 0

    loop = tqdm(loader, desc="Evaluating", leave=True, ncols=100)
    with torch.no_grad():
        for batch in loop:
            labels   = batch["label"].to(device)
            batch_in = {k: v.to(device) for k, v in batch.items()
                        if k != "label"}
            preds    = model(batch_in).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            loop.set_postfix({"acc": f"{100.0*correct/total:.2f}%"})

    acc = 100.0 * correct / total
    print(f"[Eval] Accuracy: {acc:.2f}%")
    return acc


def train_and_eval(model, train_loader, test_loader, device, epochs=10):
    train_model(model, train_loader, device, epochs=epochs)
    return evaluate(model, test_loader, device)