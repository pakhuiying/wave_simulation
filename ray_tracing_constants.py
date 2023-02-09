import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from math import ceil
import random

a_u = 3.16e-3
a_c = 1.92e-3
delta = 1 #along wind
eps = np.sqrt((3*a_u)/(4*a_c)) #cross wind
gamma = np.sqrt((delta/2)**2 + eps**2)

U = 1 # wind speed: 10 m/s
# sigma_a = a_u*U
# sigma_c = 0.003 + a_c*U
# sigma = sigma_a + sigma_c

# sigma = 0.003 + 5.12e-3*U
sigma = delta*(a_u/2*U)**0.5

r_11 = delta/(2*gamma)
r_21 = eps/gamma
r_12 = -1*r_11
r_22 = r_21*1
r_1 = np.array([r_11,r_21,0])
r_2 = np.array([r_12,r_22,0])
r_1_hat = np.array([-r_21,r_11,0])
r_2_hat = np.array([-r_22,r_12,0])

m_a = 1 #index of refraction of air
m_w = 4/3 #index of refraction of water

# wind based coordinate system
i = np.array([1,0,0])
k = np.array([0,0,1])
j = np.cross(k,i)#np.array([0,1,0])
