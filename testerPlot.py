## import packages, and conventional abbrev.
import numpy as np
import math
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d  # for 3D coordinates

x = np.outer(np.linspace(-10, 10, 30), np.ones(30))  # outer product
y = x.copy().T  # transpose
z = (1 - x) ** 2 + 100 * (y - (x ** 2)) ** 2  # define surface

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')
ax.set_title('Surface $z = (1-x)^2 + 100(y-x^2)^2 $')


# plt.show()


# define function
def f(x, y):
    value = (1 - x) ** 2 + 100 * (y - (x ** 2)) ** 2
    return value


tolerance = 0.000001
dx = 0.0001
dy = 0.0001

x0: float = -6
y0: float = -3
maxIterations = f(x0, y0)
minZ: float = 0


def gradX(x, y):
    value = (f(x + dx, y) - f(x, y)) / dx
    return value


def gradY(x, y):
    value = (f(x, y + dy) - f(x, y)) / dy
    return value


# gradient of f
def getGradient(x, y):
    gradient = [gradX(x, y), gradY(x, y)]
    return gradient


# stepSize = 1/((gradX(x0,y0) + gradY(x0,y0))/2)
stepSize = 10 / math.sqrt((gradX(x0, y0)) ** 2 + (gradY(x0, y0)) ** 2)
# stepSize = 10**-5

xPosition = [x0]
yPosition = [y0]
zPosition = [f(x0, y0)]

for a in range(maxIterations):
    scalar = stepSize
    direction = getGradient(x0, y0)
    newZ = f(x0 - stepSize * direction[0], y0 - stepSize * direction[1])
    while newZ > f(x0, y0):
        scalar = scalar / 2
        newZ = f(x0 - scalar * direction[0], y0 - scalar * direction[1])
    #print(math.fabs(f(x0, y0) - newZ))
    while newZ - f(x0, y0) > 1:
        scalar = scalar/2
        newZ = f(x0 - scalar * direction[0], y0 - scalar * direction[1])
    if math.fabs(f(x0, y0) - newZ) < tolerance:
        #print(f(x0,y0))
        #print("bruh")
        #print(newZ)
        #minZ = newZ
        break
    elif newZ < 0:
        minZ = f(x0, y0)
        break
    else:
        x0 -= scalar * direction[0]
        y0 -= scalar * direction[1]
        xPosition.append(x0)
        yPosition.append(y0)
        zPosition.append(f(x0, y0))
        minZ = newZ

print(zPosition)
print(minZ)
