import numpy as np 
import matplotlib.pyplot as plt 

def func_original(x): 
    return np.cos(x)*np.exp(-x**2)

def func_transform(t): 
    x=np.tan((np.pi/2)*t)
    derivative=(np.pi/2)*(1/np.cos((np.pi/2)*t)**2)
    return func_original(x)*derivative 

def gauss_legendre_quadrature(n): 
    if n == 2: 
        roots=np.array([-1/np.sqrt(3),1/np.sqrt(3)])
        weights=np.array([1,1])
    elif n == 3: 
        roots=np.array([-np.sqrt(3/5),0,np.sqrt(3/5)])
        weights=np.array([5/9,8/9,5/9])
    else: 
        raise ValueError('Try again for n=2 & 3 only')
    
    return roots,weights 

def gauss_legendre_integration(n): 
    roots,weights=gauss_legendre_quadrature(n)
    
    results=[]
    for i in range(n):
        results.append(weights[i]*func_transform(roots[i]))

    integral_legendre=sum(results)
    return integral_legendre

n_pt=3

result_gauss_legendre=gauss_legendre_integration(n_pt)

def gauss_hermite_quadrature(n): 
    roots,weights=np.polynomial.hermite.hermgauss(n)

    integral=np.sum(weights*np.cos(roots))

    return integral 

result_gauss_hermite=gauss_hermite_quadrature(10)

print(f'Result using Gauss-Legendre Transform Method is {result_gauss_legendre}')
print(f'Result using Gauss-Hermite Original Method is {result_gauss_hermite}')

error=abs(result_gauss_legendre-result_gauss_hermite)
print(f'Error between these method is {error}')

# Plot the functions
t_values=np.linspace(-1,1,500)
x_values=np.linspace(-5,5,500)

original_values=func_original(x_values)

plt.figure(figsize=(12,6))

plt.plot(x_values,original_values,color='blue',label=f'Original function and Domain of x∈(-∞,∞)')
plt.fill_between(x_values,original_values,color='lightblue',label=f'Area under the Original function is {result_gauss_hermite:.3f}')

transform_values=func_transform(t_values)
plt.plot(t_values,transform_values,color='green',label='Transform function and Domain of t∈[-1, 1]')
plt.fill_between(t_values,transform_values,color='lightgreen',label=f'Area under the Transform function is {result_gauss_legendre:.3f}')

plt.title('Compare of Original & Transform Function by Gauss-Hermite Method',fontsize=10)
plt.xlabel('x,t values')
plt.ylabel('Function Values')
plt.grid()
plt.legend() 
plt.show()