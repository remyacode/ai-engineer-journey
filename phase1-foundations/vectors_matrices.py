def dot_product(a, b):
    total = 0
    for i in range(len(a)):
        total = total + (a[i] * b[i])
    return total

def matrix_vector_multiply(M, v):
    result = []
    for row in M:
        result.append(dot_product(row, v))
    return result

