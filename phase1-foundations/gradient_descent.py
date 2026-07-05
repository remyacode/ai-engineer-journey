import math
from vectors_matrices import dot_product
from single_neuron import sigmoid

def train_step(x, w, b, actual, learning_rate):
    z = dot_product(x, w) + b
    predicted = sigmoid(z)

    dL_dpred = 2 * (predicted - actual)
    dsig_dz = predicted * (1 - predicted)

    new_w = []
    for i in range(len(w)):
        dz_dwi = x[i]
        gradient = dL_dpred * dsig_dz * dz_dwi
        new_w.append(w[i] - learning_rate * gradient)

    dz_db = 1
    gradient_b = dL_dpred * dsig_dz * dz_db
    new_b = b - learning_rate * gradient_b

    return new_w, new_b, predicted

new_w, new_b, predicted = train_step([1, 2], [0.5, -0.3], 0.1, 1, 0.1)
print("predicted:", predicted)
print("new_w:", new_w)
print("new_b:", new_b)