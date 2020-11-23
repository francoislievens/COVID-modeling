from scipy.stats import binom
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt


vec = np.arange(10)
vec_n = np.linspace(0, 10, 100)
y = []
y_n = []

for i in range(0, len(vec)):
    y.append(binom.pmf(k=vec[i], n=10, p=0.5))

print(np.around(0.51))

print(vec)
print(y)
plt.plot(vec, y)
plt.show()
