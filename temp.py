import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator



def exact_soluation(x_range, t_range):
    return np.exp(-(np.pi**(2))*t_range/4)*t_range*np.sin(np.pi*x_range/2)
    


x = np.linspace(0, 2, num=10)
tim = np.linspace(0,2,num =90)
# print(x)
# print(exact_soluation(x, tim))

x,t = np.meshgrid(x,tim)
u = exact_soluation(x,t)

ax = plt.axes(projection='3d')
# ax.plot_surface(x,t,u) 
ax.plot_wireframe(x,t,u,color='b') 
ax.plot_wireframe(x,t,u**(3),color='r') 
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u')
# ax.set_box_aspect((1, 1, 0.5))
# plt.title('Axes3D Plot Surface')
plt.show()