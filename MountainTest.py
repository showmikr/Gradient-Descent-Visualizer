## import packages, and conventional abbrev.
import numpy as np
import math
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d  # for 3D coordinates

x = np.outer(np.linspace(-15, 15, 80), np.ones(80))  # outer product
y = x.copy().T  # transpose

Everest = (8.8489 - 4.75) / ((0.37 / 8) * x ** 2 + (0.65 / 8) * y ** 2 + 1)
Lhotse = (8.5 - 4.75) / ((0.62 / 8) * (x - 0.85 * (5 / 8)) ** 2 + (0.62 / 8) * (y + 2.98 * (5 / 4)) ** 2 + 1)
TopLeft = (7.2 - 4.75) / ((0.21 / 8) * (x + 4.88) ** 2 + (0.36 / 4) * (y - 1.09) ** 2 + 1)
LeftRidge = (7.6 - 4.75) / (0.015 * (x + 3.825) ** 2 + 0.125 * (y + 4.5) ** 2 + 1)
RightRidge = (7.6 - 4.75) / (0.099425 * (x - 4.6285) ** 2 + 0.175 * (y + 3.5) ** 2 + 1)
RightPeak = (7.42 - 4.75) / (0.05425 * (x - 7.814) ** 2 + 0.05425 * (y + 4.357) ** 2 + 1)
TopRight = (7.7 - 4.75) / (0.05 * (x - 2.957) ** 2 + 0.05 * (y - 2.786) ** 2 + 1)

k = 3
z = (Everest ** k + Lhotse ** k + TopLeft ** k + LeftRidge ** k + RightRidge ** k + RightPeak ** k + TopRight ** k) ** (
        1 / k)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')
ax.set_title('Approximation of Mt. Everest')

#plt.show()


def f(x, y):
    Everest = (8.8489 - 4.75) / ((0.37 / 8) * x ** 2 + (0.65 / 8) * y ** 2 + 1)
    Lhotse = (8.5 - 4.75) / ((0.62 / 8) * (x - 0.85 * (5 / 8)) ** 2 + (0.62 / 8) * (y + 2.98 * (5 / 4)) ** 2 + 1)
    TopLeft = (7.2 - 4.75) / ((0.21 / 8) * (x + 4.88) ** 2 + (0.36 / 4) * (y - 1.09) ** 2 + 1)
    LeftRidge = (7.6 - 4.75) / (0.015 * (x + 3.825) ** 2 + 0.125 * (y + 4.5) ** 2 + 1)
    RightRidge = (7.6 - 4.75) / (0.099425 * (x - 4.6285) ** 2 + 0.175 * (y + 3.5) ** 2 + 1)
    RightPeak = (7.42 - 4.75) / (0.05425 * (x - 7.814) ** 2 + 0.05425 * (y + 4.357) ** 2 + 1)
    TopRight = (7.7 - 4.75) / (0.05 * (x - 2.957) ** 2 + 0.05 * (y - 2.786) ** 2 + 1)

    return (
                   Everest ** k + Lhotse ** k + TopLeft ** k + LeftRidge ** k + RightRidge ** k + RightPeak ** k + TopRight ** k) ** (
                   1 / k)


tolerance = 10 ** -5
dx = 10 ** -4
dy = 10 ** -4

x0: float = 0
y0: float = 0
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
stepSize = 1 / math.sqrt((gradX(x0, y0)) ** 2 + (gradY(x0, y0)) ** 2)
# stepSize = math.log(f(x0, y0), 10)
maxIterations = round(f(x0, y0) * 10 ** stepSize)
print(stepSize)
# stepSize = 10
# stepSize = 10**-5

xPosition = [x0]
yPosition = [y0]
zPosition = [f(x0, y0)]
counter = 0

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

print(zPosition[-1])
print(xPosition[-1])
print(yPosition[-1])
print("minZ in meters = " + str(minZ * 1000))
print("Counter: " + str(counter))
print("MaxIteration - Counter = " + str(maxIterations - counter))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X, Y, Z = np.array(xPosition), np.array(yPosition), np.array([zPosition])
ax.plot_wireframe(X, Y, Z)
plt.show()
