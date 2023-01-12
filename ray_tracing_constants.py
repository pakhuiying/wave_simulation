import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from math import ceil
import random

a_u = 3.16e-3
a_c = 1.92e-3
n = 30 #cell size of grid
delta = 1 #along wind
eps = np.sqrt((3*a_u)/(4*a_c)) #cross wind

n = 2 #order of the hexagonal grid
X = np.linspace(0,n,num=n,endpoint=False)
Y = np.arange(0,n*eps,step=eps) #np.linspace(0,n,num=n,endpoint=False)
x , y = np.meshgrid(X,Y)
x[::2,:] += 0.5 #half the width of delta
x = x.flatten()
y = y.flatten()
triang = mpl.tri.Triangulation(x, y) ## Triangulate parameter space to determine the triangles using a Delaunay triangulation

U = 10 # wind speed: 10 m/s
sigma_a = a_u*U
sigma_c = 0.003 + a_c*U
sigma = sigma_a + sigma_c

sigma = 0.003 + 5.12e-3*U

np.random.seed(seed=4)
n_dist = np.random.normal(loc=0,scale = np.sqrt(sigma),size = x.shape[0])


gamma = np.sqrt((delta/2)**2 + eps**2)
r_11 = delta/(2*gamma)
r_21 = eps/gamma
r_12 = -1*r_11
r_22 = r_21*1
r_1 = np.array([r_11,r_21,0])
r_2 = np.array([r_12,r_22,0])
r_1_hat = np.array([-r_21,r_11,0])
r_2_hat = np.array([-r_22,r_12,0])



np.random.seed(1)
p_index = random.randint(0,x.shape[0])
p_h = np.array([x[p_index],y[p_index],0])

r1 = lambda l: np.array([-2,0,0]) + l*r_1
r2= lambda l: np.array([2,0,0]) + l*r_2

p_h = r2(0.7)
solar_altitude = 70 #degrees
p_prime = p_h + np.array([0,0,np.linalg.norm(p_h)*np.tan(solar_altitude/180*np.pi)])
t = np.array([-0.5,-2,0]) # target point
xi_prime = t - p_prime
xi_h = t - p_h
xi_h = xi_h/np.linalg.norm(xi_h)
