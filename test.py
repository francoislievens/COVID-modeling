from scipy.stats import binom
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom as binom
import math


res = np.random.multinomial(5, [0.2, 0.4, 0.4])
print(res)

x = np.arange(50)
pred = 20 * 2
prob = []
logprob = []
for i in range(0, len(x)):
    n = pred
    k = x[i]
    p = 0.7 / 2

    if k > n:
        prb = binom.pmf(k=n, n=n, p=p)
        prob.append(prb)
        lgpr = np.log(prob[i])
        if lgpr == - math.inf:
            lgpr = -700
        lgpr -= np.exp(k/(n*p))
        logprob.append(lgpr)
    else:
        prob.append(binom.pmf(k=k, n=n, p=p))
        logprob.append(np.log(prob[i]))


plt.scatter(x, prob, c='blue')
plt.show()
plt.scatter(x, logprob, c='red')
plt.show()

print(prob)
print(logprob)


