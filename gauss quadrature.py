import numpy as np 

def func(x): 
    return 5*x*np.exp(-2*x)

a=eval(input('enter the value of lower limit: '))
b=eval(input('enter the value of upper limit: '))
n=eval(input('enter the value of point: '))

if n == 1: 
    weight=b-a 
    node=(b+a)/2 
    integral=weight*func(node)
    print(f'The value of integration for one point is {integral}')
elif n == 2:
    weight=(b-a)/2 
    node1=weight*(-1/(3**0.5))+(b+a)/2 
    node2=weight*(1/(3**0.5))+(b+a)/2 
    integral=weight*func(node1)+weight*func(node2)
    print(f'The value of integration for two point is {integral}')
else: 
    print('Try only for one or two point gauss quadrature only') 