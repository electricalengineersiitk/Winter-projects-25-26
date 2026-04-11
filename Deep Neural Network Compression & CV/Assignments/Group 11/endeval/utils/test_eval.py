import torch
from .training import evaluate


def test_eval(model, test_loader, device):
    """
    Full test evaluation with per-class breakdown.
    Returns overall accuracy.
    """
    model.to(device)
    model.eval()

    correct    = 0
    total      = 0
    class_correct = {}
    class_total   = {}

    with torch.no_grad():
        for batch in test_loader:
            labels   = batch["label"].to(device)
            batch_in = {k: v.to(device) for k, v in batch.items()
                        if k != "label"}

            outputs = model(batch_in)
            preds   = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total   += labels.size(0)

            # Per-class tracking
            for pred, label in zip(preds, labels):
                l = label.item()
                class_total[l]   = class_total.get(l, 0) + 1
                class_correct[l] = class_correct.get(l, 0) + (pred == label).item()

    overall_acc = 100.0 * correct / total
    print(f"\n[Test Eval] Overall Accuracy: {overall_acc:.2f}%")
    print(f"[Test Eval] Correct: {correct}/{total}")

    return overall_acc