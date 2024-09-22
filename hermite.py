import numpy as np 
import matplotlib.pyplot as plt 

def func_original(x):
    return np.exp(-x**2)*np.cos(x)

def transformation(t,lower_limit,upper_limit): 
    if upper_limit == np.inf and lower_limit == -np.inf: 
        return np.tan((np.pi/2)*t) 
    else: 
        return np.tan((np.pi/2)*t)*(upper_limit-lower_limit)/2 
    
def derivative(t,lower_limit,upper_limit): 
    if upper_limit == np.inf and lower_limit == -np.inf:
        return (np.pi/2)*(1/np.cos((np.pi/2)*t)**2)
    else: 
        return (np.pi/2)*(1/np.cos((np.pi/2)*t)**2)*(upper_limit-lower_limit)/2 
    
def func_transform(t,lower_limit,upper_limit):
    s=transformation(t,lower_limit,upper_limit)
    return func_original(s)*derivative(t,lower_limit,upper_limit)

def gauss_legendre(n):
    if n == 2: 
        roots=np.array([-1/np.sqrt(3),1/np.sqrt(3)])
        weights=np.array([1,1])
    elif n == 3: 
        roots=np.array([-np.sqrt(3/5),0,np.sqrt(3/5)])
        weights=np.array([5/9,8/9,5/9])
    else: 
        raise ValueError('Try only for n=2 or n=3 only')
    
    return roots,weights

def gauss_legendre_integration(n,lower_limit,upper_limit): 

    roots,weights=gauss_legendre(n)

    results=[]
    for i in range(n): 
        results.append(weights[i]*func_transform(roots[i],lower_limit,upper_limit))

    integral_value=sum(results)
    return integral_value 

def gauss_hermite_integration(n,lower_limit,upper_limit): 
    if lower_limit != -np.inf and upper_limit != np.inf:
        raise ValueError('Gauss-Hermite Method is only applicable for (-∞, ∞)')
    
    roots,weights=np.polynomial.hermite.hermgauss(n)

    results=[]
    for i in range(n): 
        results.append(weights[i]*func_original(roots[i]))

    results_hermite=sum(results)
    return results_hermite 

lower_limit=-np.inf 
upper_limit= np.inf 

n_pt=3 

result_legendre=gauss_legendre_integration(n_pt,lower_limit,upper_limit)

if lower_limit == -np.inf and upper_limit == np.inf:
    result_hermite = gauss_hermite_integration(n_pt, lower_limit, upper_limit)
else:
    result_hermite = None

print(f"Result using  Gauss-Legendre method: {result_legendre}")
print(f"Result using  Gauss-Hermite method: {result_hermite}") 

error=abs(result_legendre-result_hermite)
print(f'The error between these results is {error}')


t_values=np.linspace(-1,1,500)
x_values=np.linspace(-5,5,500)

original_values=func_original(x_values)

transformed_values=[]
for t in t_values: 
    transformed_values.append(func_transform(t,lower_limit,upper_limit))

plt.figure(figsize=(12,6))

plt.plot(x_values, original_values, label='Original Function (x) and Domain of x∈(-∞,∞)', color='blue')
plt.fill_between(x_values, original_values, label='Original Area under f(x)', color='lightblue')

plt.plot(t_values, transformed_values, label='Transformed Function (t)  and Domain of t∈[-1, 1]', color='green')
plt.fill_between(t_values, transformed_values, label='Transformed Function (t)', color='lightgreen')

plt.title('Comparison of Original and Transformed Functions')
plt.xlabel('x and t values')
plt.ylabel('Function value')
plt.legend()
plt.grid(True)
plt.show() 