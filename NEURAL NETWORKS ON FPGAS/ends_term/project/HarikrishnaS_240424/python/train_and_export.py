import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn import datasets

iris = datasets.load_iris()

X=iris.data
y=iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


model=tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy', #becuase we need to classify into 3 subcategories
    metrics=['accuracy']
)


model.fit(X_train, y_train, epochs=50, verbose=0)

def quantise(value):
    q = int(round(float(value) * 256))
    return max(-32768, min(32767, q))  # clamp to int16

# Convert to hex (16-bit signed)
def to_hex(val):
    return format(val & 0xFFFF, '04x')

weights=[]
biases=[]

for layer in model.layers:
    w,b=layer.get_weights()

    # Flatten and quantise
    for x in w.flatten():
        weights.append(quantise(x))
    for x in b.flatten():
        biases.append(quantise(x))


test_samples=X_test[:10]
test_labels=y_test[:10]
features=[]
test_data=[]
for i in range(10):
    for x in test_samples[i]:
        features.append(quantise(x))
    label=int(test_labels[i])
    test_data.append(features+[label])


import os
os.makedirs("weights", exist_ok=True)

with open("weights/weights.mem", "w") as f:
    for w in weights:
        f.write(to_hex(w) + "\n")


with open("weights/biases.mem", "w") as f:
    for b in biases:
        f.write(to_hex(b) + "\n")

with open("weights/test_data.mem", "w") as f:
    for row in test_data:
        for val in row:
            f.write(to_hex(val) + "\n")

predictions = model.predict(X)
pred_classes = np.argmax(predictions, axis=1)


accuracy = np.mean(pred_classes == y)
print(f"Overall accuracy: {accuracy*100:.2f}%")


