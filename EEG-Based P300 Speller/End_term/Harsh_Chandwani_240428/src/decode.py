import numpy as np

GRID = np.array([
    list("ABCDEF"),
    list("GHIJKL"),
    list("MNOPQR"),
    list("STUVWX"),
    list("YZ1234"),
    list("56789_")
])

def decode_with_reps(probs, codes, n_reps):
    probs, codes = probs[:n_reps*12], codes[:n_reps*12]
    scores = {}
    for p, c in zip(probs, codes):
        scores.setdefault(int(c), []).append(p[1])
    avg = {k: np.mean(v) for k, v in scores.items()}
    col = max(range(1, 7),  key=lambda x: avg.get(x, 0))
    row = max(range(7, 13), key=lambda x: avg.get(x, 0)) - 7
    return GRID[row][col-1]
