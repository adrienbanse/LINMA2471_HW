# LINMA2471 - Homework 1

import numpy  as np
import matplotlib.pyplot as plt

# Press the green button in the gutter to run the script.

data = np.loadtxt('HW1.txt')

x = data[:, 0]
y = data[:, 1]

plt.plot(x, y, '.r')
plt.show()
