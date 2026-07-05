def squared_error(predicted, actual):
    return (predicted - actual) ** 2
def mean_squared_error(predictions, actuals):
    total = 0
    for i in range(len(predictions)):
        total = total + squared_error(predictions[i], actuals[i])
    return total / len(predictions)
