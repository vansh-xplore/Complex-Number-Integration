import numpy as np

# Function to integrate: f(x) = exp(-x^2) * cos(x)
def func_original(x):
    return np.exp(-x**2) * np.cos(x)

# Transformation from t ∈ [-1, 1] to x ∈ (-∞, ∞)
def transformation(t):
    return np.tan((np.pi / 2) * t)  # Maps [-1, 1] to (-∞, ∞)

# Derivative of the transformation
def derivative(t):
    return (np.pi / 2) * (1 / np.cos((np.pi / 2) * t)**2)  # Derivative of tan(pi/2 * t)

# Function for integration after transformation
def integral(t):
    s = transformation(t)
    return func_original(s) * derivative(t)

# Gauss Quadrature nodes and weights for n=2 or n=3
def gauss_quadrature(n):
    if n == 2:
        roots = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
        weights = np.array([1, 1])
    elif n == 3:
        roots = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
        weights = np.array([5/9, 8/9, 5/9])
    else:
        raise ValueError("This implementation only supports n=2 or n=3.")
    return roots, weights

# Gauss Quadrature integration implementation
def gauss_quadrature_integration(n):
    roots, weights = gauss_quadrature(n)
    
    results = []
    for i in range(n):
        results.append(weights[i] * integral(roots[i]))
    
    integral_value = sum(results)
    return integral_value

# Gauss-Hermite integration using NumPy's Hermite polynomial
def gauss_hermite_integration(n):
    roots, weights = np.polynomial.hermite.hermgauss(n)
    
    results = []
    for i in range(n):
        results.append(weights[i] * func_original(roots[i]))
    
    integral_value = sum(results)
    return integral_value

n_points = 3  # Using n=3 for improved accuracy

# Perform the Gauss quadrature integration
result_quadrature = gauss_quadrature_integration(n_points)

# Perform Gauss-Hermite integration using NumPy
result_hermite = gauss_hermite_integration(n_points)

# Print results
print(f"Result using Gauss Quadrature method: {result_quadrature}")
print(f"Result using NumPy's Gauss-Hermite method: {result_hermite}")

# Calculate and print error
error = abs(result_quadrature - result_hermite)
print(f"Error between methods: {error}")
