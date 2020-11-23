from scipy.stats import binom
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt


res = np.random.multinomial(5, [0.2, 0.4, 0.4])
print(res)
