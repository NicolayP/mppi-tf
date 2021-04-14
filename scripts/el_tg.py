import numpy as np
import matplotlib.pyplot as plt

a = 4
b = 2
xc = 0
yc = 0

theta = np.linspace(0, 2*np.pi, 1000)
theta_tg = np.linspace(0.2, 2*np.pi-0.2, 10)

x = a*np.cos(theta) + xc
y = b*np.sin(theta) + yc

x_tg = a*np.cos(theta_tg) + xc
y_tg = b*np.sin(theta_tg) + yc

m = -(b**2)/(a**2)*((x_tg-xc)/(y_tg-yc))
p = y_tg - m*x_tg

x_plt = np.linspace(-4, 4, 100)
#y_plt = m*x_plt+p

print(m.shape)
print(p.shape)


plt.scatter(x, y, s=0.1, c="b")
plt.scatter(x_tg, y_tg, s=5, c="r")

for slope, inter in zip(m, p):
    plt.plot(x_plt, slope*x_plt + inter)

plt.xlim(-5, 5)
plt.ylim(-3, 3)
plt.show()
