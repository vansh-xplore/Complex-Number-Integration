import numpy as np

# Define the Laguerre polynomials using recurrence relation
def laguerre(n, x):
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 1 - x
    else:
        L_n_2 = np.ones_like(x)   # L_0(x)
        L_n_1 = 1 - x             # L_1(x)
        for k in range(2, n+1):
            L_n = ((2*k - 1 - x) * L_n_1 - (k - 1) * L_n_2) / k
            L_n_2, L_n_1 = L_n_1, L_n
        return L_n_1

# Derivative of Laguerre polynomials
def laguerre_derivative(n, x):
    return n * (laguerre(n-1, x) - laguerre(n, x)) / x

# Newton's method to find roots of the polynomial
def find_roots(n, tol=1e-12, max_iter=100):
    # Initial guesses (Chebyshev nodes as rough guesses)
    roots = np.linspace(0, n * 3, n)
    for i in range(n):
        for _ in range(max_iter):
            L = laguerre(n, roots[i])
            L_prime = laguerre_derivative(n, roots[i])
            new_root = roots[i] - L / L_prime
            if abs(new_root - roots[i]) < tol:
                roots[i] = new_root
                break
            roots[i] = new_root
    return roots

# Calculate the weights for Gauss-Laguerre quadrature
def calculate_weights(n, roots):
    weights = []
    for x_i in roots:
        L_n1 = laguerre(n+1, x_i)
        weight = x_i / ((n+1)**2 * L_n1**2)
        weights.append(weight)
    return np.array(weights)

# Gauss-Laguerre integration
def gauss_laguerre_integration(f, n):
    roots = find_roots(n)
    weights = calculate_weights(n, roots)
    
    # Perform the integration
    integral = np.sum(weights * f(roots))
    return integral

# Example: Integrating the function f(x) = x^2 over [0, ∞)
def example_function(x):
    return x**2

# Number of points
n = 5

# Perform the integration
result = gauss_laguerre_integration(example_function, n)
print(f"Approximate integral: {result}")
