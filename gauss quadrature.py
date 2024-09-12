import numpy as np 
import matplotlib.pyplot as plt 

def func(x): 
    return np.exp(x)

a=0
b=2
n=2

# Plot the function over the limits
x=np.linspace(a,b,100)
y=func(x)

# Plot the function
plt.plot(x,y,color='blue',label='exp(x)')
plt.fill_between(x,y,color='lightgreen',alpha=0.3)

if n == 1: 
    weight=b-a 
    node=(b+a)/2 
    integral=weight*func(node)
    
    # Plot the node
    plt.scatter(node,func(node),color='red',label=f'Node 1: x={node:.4}')
    
    print(f'The value of integration for one point is {integral:.4}')

elif n == 2:
    weight=(b-a)/2 
    node1=weight*(-1/(3**0.5))+(b+a)/2 
    node2=weight*(1/(3**0.5))+(b+a)/2 
    integral=weight*func(node1)+weight*func(node2)

    # Plot the nodes
    plt.scatter(node1,func(node1),color='grey',label=f'Node 1: x={node1:.4}')
    plt.scatter(node2,func(node2),color='red',label=f'Node 2: x={node2:.4}')
    
    n1=np.linspace(node1,node2,100)
    n2=func(n1)
    plt.fill_between(n1,n2,color='lightblue')
    
    print(f'The value of integration for two point is {integral:.4}')

else: 
    print('Try for one or two point gauss quadrature')  

plt.axhline(0,color='grey')
plt.axvline(0,color='grey')
plt.text(0.75,0.75,'''
 Gauss Quadrature 
Integration Method 
''')

plt.title(f'Integration of exp(x) from {a} to {b} for {n} point')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(color='grey',linewidth=0.5,linestyle='dashdot')
plt.show() 