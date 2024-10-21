import numpy as np 
import matplotlib.pyplot as plt 

def func_slope(x,y):
    return x+y 

def func_euler(func,x0,y0,xn,h):
    x_values=np.arange(x0,xn+h,h)
    y_values=[y0]

    for i in range(len(x_values)-1):
        y_new = y_values[i]+h*func(x_values[i],y_values[i])

        y_values.append(y_new)

    return x_values,y_values 

x0=0
y0=1 
xn=1
h_values = [0.025,0.05,0.1]

for h in h_values:
    x_values,y_values = func_euler(func_slope,x0,y0,xn,h)

    euler_value=y_values[-1]

    print(f"for h={h}")
    print(f"    Euler's Value is: {euler_value}")

    plt.plot(x_values,y_values,marker='^',markersize=3,label=f"for h={h}")

plt.title("Euler's Method")
plt.xlabel('x-axes')
plt.ylabel('y-axes')
plt.grid()
plt.legend()
plt.show() 