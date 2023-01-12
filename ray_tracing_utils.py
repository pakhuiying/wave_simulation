import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from math import ceil
import random
from ray_tracing_constants import *


class TIP:
    def __init__(self,k,index,s,r_1_hat,r_2_hat):
        self.delta = delta
        self.eps = eps
        self.gamma = np.sqrt((self.delta/2)**2 + self.eps**2)
        self.i = np.array([1,0,0])
        self.j = np.array([0,1,0])
        self.k = np.array([0,0,1])
        self.k = k
        self.index = index
        self.s = s
        self.p_h = p_h
        self.xi_h = xi_h
        self.coord = self.p_h + self.s*self.xi_h
        self.r_1_hat = r_1_hat
        self.r_2_hat = r_2_hat
        
        if self.k == 1:
            d_k =  np.dot(self.coord,self.r_1_hat)/np.dot(self.j,self.r_1_hat)

        elif self.k == 2:
            d_k = np.dot(self.coord,self.r_2_hat)/np.dot(self.j,self.r_2_hat)
        else:
            d_k = np.dot(self.coord, self.j)

        self.d_k = d_k

class RayTracing:
    """
    get the vertices where ray intersects the wave facet
    """
    def __init__(self,n,n_dist):
        
        self.i = np.array([1,0,0])
        self.j = np.array([0,1,0])
        self.k = np.array([0,0,1])
        self.n = n
        self.delta = delta
        self.eps = eps
        self.gamma = np.sqrt((self.delta/2)**2 + self.eps**2)
        self.p_h = p_h
        self.xi_h = xi_h
        self.r_11 = self.delta/(2*self.gamma)
        self.r_21 = self.eps/self.gamma
        self.r_12 = -1*self.r_11
        self.r_22 = self.r_21*1
        self.r_1 = np.array([self.r_11,self.r_21,0])
        self.r_2 = np.array([self.r_12,self.r_22,0])
        self.r_1_hat = np.array([-self.r_21,self.r_11,0])
        self.r_2_hat = np.array([-self.r_22,self.r_12,0])

        self.n_dist = n_dist
        self.x, self.y = self.get_hexagonal_vertices()
        self.vertices_dict = self.get_vertices_index()

    def get_hexagonal_vertices(self):
        """
        creates the hexagonal grid based on the order (n) supplied
        """
        L = 2*self.n+1 #largest number of vertices at the center
        i_0 = np.linspace(-self.n,self.n,num=L)
        x = []
        y = []
        for i in range(self.n+1): #inclusive of n
            x0 = np.linspace(-self.n+i*self.delta/2,self.n-i*self.delta/2,num=L-i)
            y0 = np.repeat(np.array([i]),x0.shape[0],axis=0)
            x.append(x0)
            y.append(y0*self.eps)
        
        y = y + ([i*-1 for i in y[1:]])
        x = x + (x[1:])
        x = np.concatenate(x)
        y = np.concatenate(y)

        plt.figure()
        plt.plot(x,y,'o')
        plt.show()
        return x,y

    def get_vertices_index(self):
        """
        returns a dict where the i,j tuple vertices and they are unique, and values are the indices corresponding to the x,y array
        so that we can search for the indices quickly using the i,j tuple information
        """
        # vertices list of the triads on the horizontal surface

        vertices_list = []
        for i,j in zip(self.x,self.y):
            vertices_list.append(np.array([i,j,0]))
        vertices_list

        #compute i and j
        vertices_dict = {} #where keys are the i,j tuple vertices and they are unique, and values are the indices corresponding to the array
        # so that we can search for the indices quickly using the i,j tuple information
        for index,v in enumerate(vertices_list):
            i = (v[0]/self.delta)
            j = (v[1]/self.eps)
            vertices_dict['{:.1f},{:.1f}'.format(i,j)] = index
        
        return vertices_dict

    def get_TIPs(self):
        s_k = lambda c,r_k_hat: (np.dot(2*c*self.eps*self.j, r_k_hat) - np.dot(self.p_h,r_k_hat))/np.dot(self.xi_h,r_k_hat)
        s_0 = lambda c: (c*self.eps - np.dot(self.p_h,self.j))/np.dot(self.xi_h, self.j)
        s_j_dict = {k: [] for k in range(0,3)} #where keys are the k family and values are the associated s_k
        
        for c in range(-self.n,self.n+1):
            for k in range(0,3):
                if k == 1:
                    r_k_hat = r_1_hat
                    s_j_dict[k].append(s_k(c,r_k_hat))
                elif k == 2:
                    r_k_hat = r_2_hat
                    s_j_dict[k].append(s_k(c,r_k_hat))
                else:
                    s_j_dict[k].append(s_0(c))
        
        #only values between 0 and s_min (minimum of the positive values) are kept as they are within the hexagon

        s_j_dict = {k:[i for i in s_j_dict[k] if i>=0] for k in s_j_dict.keys()} #filter s values >=0
        s_j_dict = {k:s_j_dict[k][:np.argmin(s_j_dict[k])+1] for k in s_j_dict.keys()} # values are sorted in descending manner, select values until the minimum positive value
        ordered_TIPs = sorted([(k,i) for k,v in s_j_dict.items() for i in v],key=lambda x: x[1]) #list of tuples, where first element represents the k family, and 2nd element represents the associated s_j value
        ordered_TIPs = [TIP(k,i,s,self.p_h,self.xi_h) for i,(k,s) in enumerate(ordered_TIPs)]
        return ordered_TIPs

    def TIP_vertices(self,TIP_0,TIP_1):
        """
        TIP_0 (class TIP)
        TIP_1 (class TIP)
        returns the triad facet vertices where both TIPs intersect the wave facet
        >>> (a,b,c) = TIP_vertices(TIP1,TIP2) 
        """
        if TIP_0.k > TIP_1.k:
            TIP_0, TIP_1 = TIP_1, TIP_0

        triad_case = int(TIP_0.k + TIP_1.k)

        if triad_case == 3: #r1r2
            d_2 = TIP_1.d_k
            d_1 = TIP_0.d_k
            h = (d_2 - d_1)*self.delta/(4*self.eps)
            a_1 = h
            a_2 = 0.5*(d_1 + d_2)
            a = np.array([a_1,a_2,0])
            lambda_k_a_list = []
            lambda_k_list = []
            for i in range(1,3): #k=1,2
                if i == 1:
                    d_k = d_1
                    r_k = self.r_1
                    y_k = TIP_0.coord
                elif i == 2:
                    d_k = d_2
                    r_k = self.r_2
                    y_k = TIP_1.coord

                lambda_k_a = (a - np.dot(d_k,self.j))
                lambda_k_a = np.dot(lambda_k_a,r_k)
                lambda_k_a_list.append(lambda_k_a)
                lambda_k = (y_k -np.dot(d_k,self.j))
                lambda_k = np.dot(lambda_k,r_k)
                lambda_k_list.append(lambda_k)
            
            sign_list = [] #sign for (lambda_1 - lambda_1_a), or (lambda_2 - lambda_2_a)
            for i in range(0,len(lambda_k_list)):
                if lambda_k_list[i] - lambda_k_a_list[i] < 0:
                    sign = -1
                else:
                    sign = 1
                sign_list.append(sign)
            
            b_i = (a_1 + sign_list[0]*self.delta/2)
            b_j = (a_2 + sign_list[0]*self.eps)
            b = np.array([b_i,b_j,0])

            c_i = (a_1 - sign_list[1]*self.delta/2)
            c_j = (a_2 + sign_list[1]*self.eps)
            c = np.array([c_i,c_j,0])

            return (a,b,c)

        elif triad_case == 1:
            d_0 = TIP_0.d_k
            d_1 = TIP_1.d_k
            h = (d_0 - d_1)*self.delta/(2*self.eps)
            a_1 = h
            a_2 = d_0
            a = np.array([a_1,a_2,0])
            y_1 = TIP_1.coord
            lambda_0 = np.dot(TIP_0.coord,self.i)
            lambda_0_a = np.dot(a,self.i)

            lambda_1_a = (a - np.dot(d_1,self.j))
            lambda_1_a = np.dot(lambda_1_a,self.r_1)

            lambda_1 = (y_1 -np.dot(d_1,self.j))
            lambda_1 = np.dot(lambda_1,self.r_1)

            if lambda_1 - lambda_1_a < 0:
                sign_1 = -1
            else:
                sign_1 = 1
            
            if lambda_0 - lambda_0_a < 0:
                sign_0 = -1
            else:
                sign_0 = 1
            
            b_i = (a_1 + sign_1*self.delta/2)
            b_j = (a_2 + sign_1*self.eps)
            b = np.array([b_i,b_j,0])

            c_i = (a_1 + sign_0*self.delta)
            c_j = a_2
            c = np.array([c_i,c_j,0])
            return (a,b,c)

        else: #if case = 2
            d_2 = TIP_1.d_k
            d_0 = TIP_0.d_k
            h = (d_2 - d_0)*self.delta/(2*self.eps)
            a_1 = h
            a_2 = d_0
            a = np.array([a_1,a_2,0])
            y_2 = TIP_1.coord

            lambda_0 = np.dot(TIP_0.coord,self.i)
            lambda_0_a = np.dot(a,self.i)

            lambda_2_a = (a - np.dot(d_2,self.j))
            lambda_2_a = np.dot(lambda_2_a,self.r_2)

            lambda_2 = (y_2 - np.dot(d_2,self.j))
            
            lambda_2 = np.dot(lambda_2,self.r_2)

            if lambda_2 - lambda_2_a < 0:
                sign_2 = -1
            else:
                sign_2 = 1
            
            if lambda_0 - lambda_0_a < 0:
                sign_0 = -1
            else:
                sign_0 = 1

            b_i = (a_1 - sign_2*self.delta/2)
            b_j = (a_2 + sign_2*self.eps)
            b = np.array([b_i,b_j,0])

            c_i = (a_1 + sign_0*self.delta)
            c_j = a_2
            c = np.array([c_i,c_j,0])
            return (a,b,c)

    def determine_nodes(self,intersect_vertices_list):
        """ 
        intersect_vertices_list (list of np.arrays): Triad Vertices that intersect with ray path
        vertices_list (list of np.arrays): Triad vertices of the hexagonal domain
        n_dist (list of float): random heights at the triad vertices
        """
        nodes_list = []
        for triplets in intersect_vertices_list:
            nodes = []
            for a in triplets:
                i = (a[0]/self.delta)
                j = (a[1]/self.eps)
                k = '{:.1f},{:.1f}'.format(i,j)
                if k in self.vertices_dict.keys():
                    index = self.vertices_dict[k]
                    n = self.n_dist[index]
                    v = a + np.array([0,0,n])
                    nodes.append(v)
            if len(nodes) == 3: #remove vertices tha are outside the hexagonal domain
                nodes_list.append(nodes)

        return nodes_list
    

    def get_normal_facet(v_list):
        """
        v_list (list): nodes of v_1,v_2,v_3 (in order)
        """
        n1 = np.cross((v_list[2] - v_list[0]),(v_list[1] - v_list[0]))
        n1 = n1/np.linalg.norm(n1)

        if np.dot(n1,np.array([0,0,1])) < 0:
            sign = -1
        else:
            sign = 1

        n1 = sign*n1
        return n1