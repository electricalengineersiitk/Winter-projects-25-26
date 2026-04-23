import numpy as np
weights = np.random.randn(32)
biases = np.random.randn(8)

def quantize(x):
    q = int(round(x * 256))
    return max(-32768, min(32767, q))

# Save weights
with open("../weights/weights.mem", "w") as f:
    for w in weights:
        f.write(f"{quantize(w) & 0xFFFF:04x}\n")

# Save biases
with open("../weights/biases.mem", "w") as f:
    for b in biases:
        f.write(f"{quantize(b) & 0xFFFF:04x}\n")

print("Weights and biases exported successfully")
