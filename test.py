import numpy as np

mu1 = [1, 0]
mu2 = [0, 1.5]
sigma1 = np.matrix('1 0.75; 0.75 1')
sigma2 = np.matrix('1 0.75; 0.75 1')

#a = np.random.multivariate_normal(mu1, sigma1, 500)

a = np.concatenate((np.random.multivariate_normal(mu1, sigma1, 20), np.random.multivariate_normal(mu1, sigma1, 20)))
l = np.concatenate((np.zeros((20,1)), np.ones((20,1))))
d = np.append(a, l, 1)
print(d.shape)