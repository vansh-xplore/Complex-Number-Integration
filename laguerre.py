import numpy as np

# New function to integrate
def func_original(x):
    return np.cos(x) * np.exp(-x)  

# Transformation function: mapping t ∈ [-1, 1] to x ∈ [lower_limit, upper_limit]
def transformation(t, lower_limit, upper_limit):
    if upper_limit == np.inf:
        return lower_limit + (1 + t) / (1 - t)  # Maps [-1, 1] to [lower_limit, ∞)
    else:
        return (upper_limit + lower_limit) / 2 + (upper_limit - lower_limit) / 2 * t

# Derivative of the transformation 
def derivative(t, lower_limit, upper_limit):
    if upper_limit == np.inf:
        return 2 / (1 - t) ** 2
    else:
        return (upper_limit - lower_limit) / 2

# Function for integration after transformation
def integral(t, lower_limit, upper_limit):
    s = transformation(t, lower_limit, upper_limit)
    return func_original(s) * derivative(t, lower_limit, upper_limit)

# Gauss-Legendre nodes and weights for n=2 or n=3
def gauss_legendre(n):
    if n == 2:
        roots = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
        weights = np.array([1, 1])
    elif n == 3:
        roots = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
        weights = np.array([5/9, 8/9, 5/9])
    else:
        raise ValueError("This implementation only supports n=2 or n=3.")
    return roots, weights

# Gauss-Legendre quadrature implementation
def gauss_legendre_integration(n, lower_limit, upper_limit):
    roots, weights = gauss_legendre(n)
    
    results = []  # Renamed from contributions
    for i in range(n):
        results.append(weights[i] * integral(roots[i], lower_limit, upper_limit))
    
    integral_value = sum(results)
    return integral_value

# Gauss-Laguerre quadrature implementation using NumPy
def gauss_laguerre_np(n, lower_limit, upper_limit):
    if lower_limit != 0 or upper_limit != np.inf:
        raise ValueError("Gauss-Laguerre quadrature is only applicable for [0, ∞)")

    roots, weights = np.polynomial.laguerre.laggauss(n)
    
    results = []  # Renamed from contributions
    for i in range(n):
        results.append(weights[i] * func_original(roots[i]))
    
    result = sum(results)
    return result

lower_limit = 0
upper_limit = np.inf

n_points = 3  # Using n=3 for improved accuracy

# Custom Gauss-Legendre quadrature after transformation
result_legendre = gauss_legendre_integration(n_points, lower_limit, upper_limit)

# Gauss-Laguerre quadrature (only valid for [0, ∞))
if lower_limit == 0 and upper_limit == np.inf:
    result_laguerre_np = gauss_laguerre_np(n_points, lower_limit, upper_limit)
else:
    result_laguerre_np = None

# Print results
print(f"Result using custom Gauss-Legendre method: {result_legendre}")
if result_laguerre_np is not None:
    print(f"Result using NumPy's Gauss-Laguerre method: {result_laguerre_np}")

# Calculate and print error if applicable
if result_laguerre_np is not None:
    error = abs(result_legendre - result_laguerre_np)
    print(f"Error between methods: {error}") 