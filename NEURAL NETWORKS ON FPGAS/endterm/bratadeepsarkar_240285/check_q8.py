import numpy as np
import os

# Read memory files
def read_mem(filepath):
    vals = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                # convert 16-bit hex two's complement to signed int
                v = int(line.strip(), 16)
                if v >= 32768:
                    v -= 65536
                vals.append(v)
    return vals

w_hidden = read_mem('weights/weights_hidden.mem')
b_hidden = read_mem('weights/biases_hidden.mem')
w_output = read_mem('weights/weights_output.mem')
b_output = read_mem('weights/biases_output.mem')
test_data = read_mem('weights/test_data.mem')

# sample 0 features
x = test_data[0:4]
print("Input features:", x)

# Hidden layer
hidden = []
for n in range(8):
    acc = 0
    for i in range(4):
        acc += x[i] * w_hidden[n*4 + i]
    # add bias shifted
    acc += (b_hidden[n] << 8)
    
    # relu and truncate
    if acc < 0:
        out = 0
    else:
        out = (acc >> 8) & 0xFFFF
        if out >= 32768: out -= 65536 # actually it can't be negative since it's from relu
    hidden.append(out)

print("Hidden layer outputs:", [hex(h) for h in hidden])

# Output layer
out = []
for n in range(3):
    acc = 0
    print(f"Output Neuron {n}:")
    for i in range(8):
        prod = hidden[i] * w_output[n*8 + i]
        acc += prod
        print(f"  Cycle {i}: in={hex(hidden[i])} * w={hex(w_output[n*8 + i])} = {hex(prod)}  | acc={hex(acc)}")
    acc += (b_output[n] << 8)
    print(f"  + bias<<8 = {hex(acc)}")
    
    # ReLU (wait! The hardware uses ReLU for output layer too!)
    if acc < 0:
        o = 0
    else:
        o = (acc >> 8) & 0xFFFF
    out.append(o)

print("Output layer outputs:", [hex(o) for o in out])
