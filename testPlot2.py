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


# define function
def f(x, y):
    value = (1 - x) ** 2 + 100 * (y - (x ** 2)) ** 2
    return value


tolerance = 0.00001
dx = 10 ** -4
dy = 10 ** -4

x0: float = -6
y0: float = -3
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


stepSize = 1 / math.sqrt((gradX(x0, y0)) ** 2 + (gradY(x0, y0)) ** 2)
maxIterations = round(f(x0, y0) * 10 ** stepSize)
xPosition = [x0]
yPosition = [y0]
zPosition = [f(x0, y0)]
counter = 0
negativeReduce = 0

for a in range(maxIterations):
    counter += 1
    direction = getGradient(x0, y0)
    newZ = f(x0 - stepSize * direction[0], y0 - stepSize * direction[1])
    while newZ > f(x0, y0):
        print("reduced stepSize")
        stepSize = stepSize / 2
        newZ = f(x0 - stepSize * direction[0], y0 - stepSize * direction[1])
    if math.fabs(f(x0, y0) - newZ) < tolerance:
        minZ = newZ
        break
    else:
        x0 -= stepSize * direction[0]
        y0 -= stepSize * direction[1]
        xPosition.append(x0)
        yPosition.append(y0)
        zPosition.append(f(x0, y0))
        minZ = newZ

print(zPosition)
print(minZ)
print("Max Iterations: " + str(maxIterations))
print("Counter: " + str(counter))
print("Reductions made b/c newZ < 0: " + str(negativeReduce))
print(xPosition[-1])
print(yPosition[-1])
print(f(0.3849540088510141, 0.14519276997997535))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X, Y, Z = np.array(xPosition), np.array(yPosition), np.array([zPosition])
ax.plot_wireframe(X, Y, Z)
plt.show()
