
import matplotlib.pyplot as plt

world_area = (1500, 1000)

start = (100, 500)
goal = (1200, 500)

plt.plot([0,0,1500,1500,0],[0,1000,1000,0,0])
plt.scatter(start[0], start[1])
plt.scatter(goal[0], goal[1])

plt.show()



