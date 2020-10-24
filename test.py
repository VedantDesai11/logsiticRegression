import numpy as np


a = np.zeros((5))
b = np.array([0,0,0,1,1])
t = np.equal(a,b)
c = np.count_nonzero(t)
a = np.where(b == 1)[0]

b[a] = 4
print(b)

for i in range(0, 100, 5):
	threshold = i / 100
	print(threshold)


