## import packages, and conventional abbrev.
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d # for 3D coordinates
import math

x = np.outer(np.linspace(-10, 10, 30), np.ones(30)) # outer product
y = x.copy().T # transpose
z = (1 - x)**2 + 100*(y - (x**2))**2 # define surface

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_surface(x, y, z,cmap='viridis', edgecolor='none')
ax.set_title('Surface $z = (1-x)^2 + 100(y-x^2)^2 $')
#plt.show()

# define function
def f(x,y):
    value = (1-x)**2 + 100*(y-(x**2))**2
    return value

tolerance = 0.000001
dx = 0.0001
dy = 0.0001

x0 = -6
y0 = -3
maxIterations = f(x0,y0)
min = 0

def gradX(x,y):
    value = (f(x+dx,y) - f(x,y))/(dx)
    return value
def gradY(x,y):
    value = (f(x,y+dy) - f(x,y))/(dy)
    return value

# negative gradient of f
def getGradient(x,y):
    gradient = [gradX(x,y) , gradY(x,y)]
    return gradient

#stepSize = 1/((gradX(x0,y0) + gradY(x0,y0))/2)
stepSize = 10/math.sqrt((gradX(x0,y0))**2 + (gradY(x0,y0))**2)
#stepSize = 10**-5

for a in range(maxIterations):
    direction = getGradient(x0, y0)
    newPosition = f(x0 - stepSize*direction[0], y0 - stepSize*direction[1])
    if math.fabs(f(x0, y0) - newPosition) < tolerance:
        min = newPosition
        break
    else:
        x0 -= stepSize*direction[0]
        y0 -= stepSize*direction[1]

min = newPosition
print(min)
print(f(1,1))
print(f(-6 + stepSize*gradX(-6,-3), -3 + stepSize*gradY(-6,-3)))
print(f(-6,-3))
print(gradX(-6,-3))
print(gradY(-6,-3))