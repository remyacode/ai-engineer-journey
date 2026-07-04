import math
from vectors_matrices import dot_product

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def neuron(x, w, b):
    z = dot_product(x, w) + b
    return sigmoid(z)