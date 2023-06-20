import numpy as np
import numba as nb
import copy
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def exact_solution(x, t):
    return np.exp(-((np.pi**2) * t)/4) * np.sin((np.pi * x)/2)


h = 500
h_sep = 2/h
k = 100
k_sep = 100/k
x = np.linspace(0, 2, num=h)
tim = np.linspace(0,30,num = k)
x,t = np.meshgrid(x,tim)

def bound_condition(Lx, Lt):
    u = np.zeros([len(Lx)])
    for i in range(len(Lx)):
        if Lx[i] == 0 or Lx[i] == 2:
            u[i]==0
        elif Lt[i] == 0:
            u[i] = np.sin(np.pi*Lx[i]/2)
    return u 


def crank_Ametrix(x_len):
    x_len = x_len-2
    lamb = k_sep/(2*h_sep**(2))
    matrix = np.zeros([x_len,x_len])
    for i in range(x_len):
        matrix[i][i] = lamb +1
        if i ==(x_len-1):
            continue
        matrix[i][i+1] = -lamb/2
        matrix[i+1][i] = -lamb/2
    return matrix

def crank_Bmetrix(x_len):
    x_len = x_len-2
    lamb = k_sep/(2*h_sep**(2))
    matrix = np.zeros([x_len,x_len])
    for i in range(x_len):
        matrix[i][i] = 1 - lamb 
        if i ==(x_len-1):
            continue
        matrix[i][i+1] = lamb/2
        matrix[i+1][i] = lamb/2
    return matrix  

@nb.jit()
def SOR(A_matrix,B_matrix,w = 1):
    N= 100000
    TOL = 1e-6
    k = 1
    n = A_matrix.shape[0]
    init_x = np.zeros_like(B_matrix,dtype="float64")
    x = copy.deepcopy(init_x)
    while(k<N):
        x_p = copy.deepcopy(x)
        for i in range(n):
            summ = 0.0
            for j in range(n):
                if j==i:
                    continue
                summ += A_matrix[i][j] * x[j]
                
            x[i] = w*(-summ + B_matrix[i])/A_matrix[i][i]

            x[i] +=(1-w)*x_p[i]
        esp = escape(x_p,x,TOL)[0]
        if esp:
            break
        k+=1
    def escape(old_x,new_x,TOL):
        if np.max(np.abs(old_x - new_x)) < TOL:
            return True,np.max(np.abs(old_x - new_x))
        return False,np.max(np.abs(old_x - new_x))
    return x



u2 = []
for i in range(k): # by t
    if i == 0:
        u2.append(bound_condition(x[0],t[0]))
    else:
        u2.append(copy.deepcopy(u2[i-1]))
        
        B = u2[i][1:-1]
        B = B @ crank_Bmetrix(h)
        u2[i][1:-1] = SOR(crank_Ametrix(h),B)

u2 = np.array(u2)

u = exact_solution(x,t)
ax = plt.axes(projection='3d')
# ax.plot_wireframe(x,t,u,color='g') 

ax.plot_surface(x,t,u, cmap='viridis') 
ax.plot_wireframe(x,t,u2,color='r') 

ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u')

plt.show()

print(np.mean((u-u2)**2))