import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow import keras

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = keras.Sequential()
model.add(keras.layers.Dense(8, activation='relu', input_shape=(4,)))
model.add(keras.layers.Dense(3, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, verbose=0)

weights = model.get_weights()

values = []
for w in weights:
    values.extend(w.flatten())

def to_fixed(x):
    val = int(round(x * 256))
    if val > 32767:
        val = 32767
    if val < -32768:
        val = -32768
    return val

fixed_vals = [to_fixed(v) for v in values]

f = open("weights/weights.mem", "w")
for v in fixed_vals:
    f.write(format(v & 0xFFFF, '04x') + "\n")
f.close()

f = open("weights/biases.mem", "w")
for v in fixed_vals:
    f.write(format(v & 0xFFFF, '04x') + "\n")
f.close()

f = open("weights/test_data.mem", "w")

for i in range(10):
    for val in X_test[i]:
        q = to_fixed(val)
        f.write(format(q & 0xFFFF, '04x') + "\n")
    f.write(str(y_test[i]) + "\n")

f.close()

print("mem files created")