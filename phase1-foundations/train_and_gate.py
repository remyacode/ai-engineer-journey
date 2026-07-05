from gradient_descent import train_step
from vectors_matrices import dot_product
from single_neuron import sigmoid
from loss_functions import mean_squared_error

X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [0, 0, 0, 1]

w = [0.1, 0.1]
b = 0.1
learning_rate = 0.5
epochs = 1000

for epoch in range(epochs):
    predictions = []
    for i in range(len(X)):
        w, b, predicted = train_step(X[i], w, b, Y[i], learning_rate)
        predictions.append(predicted)

    if epoch % 100 == 0:
        loss = mean_squared_error(predictions, Y)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

print("Final weights:", w)
print("Final bias:", b)