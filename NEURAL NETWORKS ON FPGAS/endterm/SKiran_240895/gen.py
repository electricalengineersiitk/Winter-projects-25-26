import tensorflow
from tensorflow.keras import layers
import numpy as np
import sklearn
from sklearn.datasets import load_iris

data = load_iris()
X = data.data
y = data.target
caler= sklearn.preprocessing.MinMaxScaler()
X = caler.fit_transform(X)
modl = tensorflow.keras.models.Sequential([
    layers.Dense(4),
    layers.Dense(8,activation='relu'),
    layers.Dense(3,activation='softmax')
])
modl.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
modl.fit(X,y,epochs=7)
weights_biases = modl.get_weights()
weights = []
biases = []
test_data = []
for i in range(len(weights_biases)):
    if not(i%2):
        weights.append(np.int16(np.round(weights_biases[i]*256)))
    else:
        biases.append(np.int16(np.round(256*weights_biases[i])))
for i in range(10):
    test_data.append([np.int16(np.round(256*X[i])),y[i]])
    import pickle
main_path1 = r'your-path\weights\weights.mem'
main_path2 = r'your-path\weights\biases.mem'
main_path3 = r'your-path\weights\test_data.mem'
def to_bin16(val):
    fixed_val = int(round(val * (2**8)))
    if fixed_val < 0:
        fixed_val = (1 << 16) + fixed_val
    return format(fixed_val & 0xFFFF, '016b')

with open(main_path1, 'w') as file:
    for layer_weights in weights:
        shape = np.shape(layer_weights)
        for j in range(shape[1]):
            for k in range(shape[0]):
                bin_str = to_bin16(layer_weights[k][j])
                file.write(bin_str + "\n")    
            #print(i[k][j])
with open(main_path2,'w') as file:
    for i in biases:
        shape = np.shape(i)
        for k in range(shape[0]):
            file.write(to_bin16(i[k])+"\n")
print('done')