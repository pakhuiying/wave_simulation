import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from math import ceil
import json
from os.path import join, exists
from os import mkdir,listdir
from tqdm import tqdm
from collections import deque #to implement stack
# import scripts
import ray_tracing_constants as rtc
from option import args
from hierarchical_ray import Forest

class BoundaryPoints:
    def __init__(self,HD, solar_altitude,solar_azimuth,step):
        """
        HD (HexagonalDomain class)
        solar_altitude (float): in degrees, angle between horizontal surface to location of "sun"
        solar_azimuth (float): in degrees, angle between location of sun and +ve i direction
        step (float): subdivisions to divide the slanted_length/horizontal_length
        
        # n (int): order of HD
        # n_max (float): max height of wave facet
        # zenith (float): angle (rad) bounded by ray and vertical
        # d (float): projected length of ray on the horizontal
        # s (np.array): point on the vector
        # l (float): scaling factor of vector
        # r_1 is pointing towards +ve i and +ve j direction
        # r_2 is pointing towards -ve i and +ve j direction
        # i_vector is point towards +ve i
        """
        self.n = HD.n
        self.n_max = HD.n_max
        self.solar_altitude = solar_altitude/180*np.pi #in rad
        self.solar_azimuth = solar_azimuth/180*np.pi #in rad
        self.solar_zenith = np.pi/2 - self.solar_altitude #in rad
        self.d = np.tan(self.solar_zenith)*self.n_max
        self.delta = rtc.delta
        self.eps = rtc.eps
        self.gamma = rtc.gamma
        self.corner_points = HD.corner_points
        # get points along the vector
        self.r1_vector = lambda s,l: s + l*rtc.r_1
        self.r2_vector = lambda s,l: s + l*rtc.r_2
        self.i_vector = lambda s,l: s+ l*np.array([1,0,0])
        
        self.step = step
        self.slanted_length = self.n*self.gamma
        self.horizontal_length = self.n*self.delta
        self.slanted_division = np.arange(0,self.slanted_length,self.step)
        self.horizontal_division = np.arange(0,self.horizontal_length,self.step)
        self.xi_prime = self.get_xi_prime()
        
    def get_boundary_points(self):
        """
        returns points_dict (dict): where keys (int) corresponds to the index of corner points of HD, from x0 to x3 in a cw direction,
            and values (list of np.arrays) correspond the the coordinate points generated along the r1,r,r2 vectors (on all surface boundaries of the HD)
            Do refer to the labelling of the corner points on the HD
        """
        
        # random_index = random.randint(0,len(self.slanted_division)-1) #index to find the length along the vector
        # x0_x2 = self.r1_vector(self.corner_points[1],self.slanted_division[random_index])
        # x2_x4 = self.i_vector(self.corner_points[2],self.horizontal_division[random_index])
        # x4_x6 = self.r2_vector(self.corner_points[4],-self.slanted_division[random_index])
        # x6_x5 = self.r1_vector(self.corner_points[6],-self.slanted_division[random_index])
        # x5_x3 = self.i_vector(self.corner_points[5],-self.horizontal_division[random_index])
        # x3_x0 = self.r2_vector(self.corner_points[3],self.slanted_division[random_index])

        points_dict = {i:[] for i in range(1,7)} #which corresponds to the corner points of HD, from x0 to x3 in a cw direction
        for i,v in enumerate([self.r1_vector, self.i_vector, self.r2_vector]):
            coord_start_index = i+1
            coord_start_index_rev = 7 - coord_start_index
            for s_l in range(len(self.slanted_division)): #iterate across all the subdivisions along the length
                if i%2 == 0: #if either the r1 or r2 vector
                    points_dict[coord_start_index].append(v(self.corner_points[coord_start_index],self.slanted_division[s_l]))
                    points_dict[coord_start_index_rev].append(v(self.corner_points[coord_start_index_rev],-self.slanted_division[s_l]))
            for h_l in range(len(self.horizontal_division)): #iterate across all the subdivisions along the length
                if i%2 != 0: #if it's the i_vector
                    points_dict[coord_start_index].append(v(self.corner_points[coord_start_index],self.horizontal_division[h_l]))
                    points_dict[coord_start_index_rev].append(v(self.corner_points[coord_start_index_rev],-self.horizontal_division[h_l]))
        
        return points_dict

    def get_points_within_HD(self):
        """
        step (float): distance between adjacent points
        returns points within the HD with height at n_max (highest wave facet) to ensure that they intersect the wave facet
        """
        ref_point = np.array([-self.n*self.delta,0,0]) #x0 point
        x_dist = lambda x1,x2: abs(x1[0]-x2[0]) # horizontal x distance between two points, x1, x2 represents the np.arrays of coord
        base_of_parallelogram = lambda d: self.n*self.gamma + 2*self.gamma/self.delta*d #where d represents the horizontal x distance
        # if distance is zero it is just the length x0 to x3
        BP = self.get_boundary_points()
        points_list = []
        
        for s in BP[1]:
            d = x_dist(s,ref_point)
            L = base_of_parallelogram(d)
            for l in np.arange(0,L,step=self.step):
                p = self.r2_vector(s,-l)
                points_list.append(np.array([p[0],p[1],self.n_max]))
                points_list.append(np.array([-p[0],-p[1],self.n_max]))
        return points_list

    def get_xi_prime(self):
        """
        returns unit xi_prime, directed downwards, towards azimuth
        """
        # location of sun
        theta_s = self.solar_zenith 
        phi_s = self.solar_azimuth
        # direction of where ray is travelling
        theta = np.pi - theta_s
        phi = (phi_s + np.pi)%(2*np.pi) #modulo 360
        self.ray_zenith = theta
        self.ray_azimuth = phi
        xi_prime = np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
        # xi_prime = np.array([self.d*np.cos(self.azimuth),self.d*np.sin(self.azimuth),-self.n_max]) # target where the ray hits the horizontal surface i.e. distance from p_h to target
        t = xi_prime/np.linalg.norm(xi_prime) # ray is directed
        # lambda_t = lambda t,p: t + np.array([p[0],p[1],0])
        # points_within_HD = self.get_points_within_HD()
        # unit_xi_prime = [(lambda_t(t,p)-p) for p in points_within_HD]
        return t


    
class TIP:
    """
    stores the attributes of TIPs
    k (int): k family to identify if TIP lies on the r_0, r_1 or r_2 line
    s (float): scalar distance from p_h along xi_h
    index (int): the order (in ascending order) of TIPs
    coord (np.array): location of TIPs w.r.t origin
    """
    def __init__(self,k,index,s,p_h,xi_h):
        self.delta = rtc.delta
        self.eps = rtc.eps
        self.r_1_hat = rtc.r_1_hat
        self.r_2_hat = rtc.r_2_hat
        self.gamma = rtc.gamma
        self.i = rtc.i#np.array([1,0,0])
        self.j = rtc.j#np.array([0,1,0])
        self.k = rtc.k#np.array([0,0,1])
        self.k = k
        self.index = index
        self.s = s
        self.p_h = p_h
        self.xi_h = xi_h
        self.coord = self.p_h + self.s*self.xi_h
        
        if self.k == 1:
            d_k =  np.dot(self.coord,self.r_1_hat)/np.dot(self.j,self.r_1_hat)

        elif self.k == 2:
            d_k = np.dot(self.coord,self.r_2_hat)/np.dot(self.j,self.r_2_hat)
        else:
            d_k = np.dot(self.coord, self.j)

        self.d_k = d_k

class WaveFacet:
    """
    obtains the attributes of wave facet
    nodes (tuple of np.array): triple tuple of array of nodes that define the triad vertices
    norm (np.array): (3,) array of unit vector of the normal of the wave triad facets
    """
    def __init__(self,nodes):
        self.nodes = nodes
        self.norm = self.get_norm()
        self.tilt = self.get_tilt()
    
    def get_norm(self):
        n1 = np.cross((self.nodes[2] - self.nodes[0]),(self.nodes[1] - self.nodes[0]))
        n1 = n1/np.linalg.norm(n1)

        if np.dot(n1,np.array([0,0,1])) < 0:
            sign = -1
        else:
            sign = 1

        n1 = sign*n1
        return n1

    def get_tilt(self):
        """
        get average tilt of the wave facet's unit normal n from the vertical k (aka beta)
        """
        return np.arccos(np.dot(self.norm,np.array([0,0,1])))

class DaughterRay:
    """
    theta_r (float): angle of incidence = reflectance (in rad)
    theta_t (float): angle of refraction (in rad)
    xi_r (np.array): vector of reflectance
    xi_t (np.array): vector of transmittance
    WF (WaveFacet class): wave facet where incident ray intercepts with WaveFacet
    returns the attributes of a daughter ray
    """
    def __init__(self,p_prime,xi_prime,theta_r,theta_t,xi_r,xi_t,WF):
        self.p_prime = p_prime
        self.xi_prime = xi_prime
        self.theta_r = theta_r
        self.theta_t = theta_t
        self.xi_r = xi_r
        self.xi_t = xi_t
        self.fresnel_reflectance = self.get_fresnel_reflectance()
        self.fresnel_transmittance = 1- self.fresnel_reflectance
        self.WF = WF

    def get_fresnel_reflectance(self):
        """
        DR (DaughterRay class)
        """
        if (isinstance(self.theta_t,float)) and (isinstance(self.xi_t,np.ndarray)): #(self.theta_t is not None) and (self.xi_t is not None):
            reflectance_theta_prime = 0.5*((np.sin(self.theta_r - self.theta_t)/np.sin(self.theta_r+self.theta_t))**2 + (np.tan(self.theta_r - self.theta_t)/np.tan(self.theta_r+self.theta_t))**2)
            return reflectance_theta_prime #r(\xi_prime \cdot n)
        else: #if Total Internal Reflection (TIR) occurs
            return 1
    


class HexagonalDomain:
    """
    computes the hexagonal domain 
    """
    def __init__(self,n):
        """
        n (int): order of hexagonal grid
        p_h (np.array): point where projected light ray enters the hexagonal domain
        p_prime (np.array): point where light ray enters the hexagonal domain
        xi_h (np.array): a unit vector of projected ray path 
        xi_prime (np.array): a vector of ray path
        vertices_dict (dict): where keys are the vertices coordinates (str); values are the unique index (int) for each vertex
        nodes_dict (dict): where keys are the unique index (int) in vertices_dict; and values are the (3,) np.array of the nodes coordinate 
        """
        self.i = rtc.i#np.array([1,0,0])
        self.j = rtc.j#np.array([0,1,0])
        self.k = rtc.k#np.array([0,0,1])
        self.n = n
        self.delta = rtc.delta
        self.eps = rtc.eps
        self.gamma = rtc.gamma

        # self.p_h = p_h
        # self.xi_h = xi_h
        # self.p_prime = p_prime
        # self.xi_prime = xi_prime

        self.r_11 = self.delta/(2*self.gamma)
        self.r_21 = self.eps/self.gamma
        self.r_12 = -1*self.r_11
        self.r_22 = self.r_21*1
        self.r_1 = np.array([self.r_11,self.r_21,0])
        self.r_2 = np.array([self.r_12,self.r_22,0])
        self.r_1_hat = np.array([-self.r_21,self.r_11,0])
        self.r_2_hat = np.array([-self.r_22,self.r_12,0])

        self.x, self.y = self.get_hexagonal_vertices()
        self.vertices_dict = self.get_vertices_index()
        self.corner_points = self.get_corner_points()

    def get_hexagonal_vertices(self):
        """
        creates the hexagonal grid based on the order (n) supplied
        """
        L = 2*self.n+1 #largest number of vertices at the center
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

        # plt.figure()
        # plt.plot(x,y,'o')
        # plt.show()
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

    def get_nodes(self,n_dist):
        """
        n_dist (list of np.array): height of wave surface at each vertex
        get the nodes (x,y,z) from the vertices (x,y,0)
        """
        nodes_dict = {k: None for k in self.vertices_dict.values()}
        for index in self.vertices_dict.values():
            z = n_dist[index]
            nodes_dict[index] = np.array([self.x[index],self.y[index],z])
        
        self.n_min = np.min(n_dist) #record the min and max of nodes to check if ray is within this vertical range
        self.n_max = np.max(n_dist)
        self.n_dist = n_dist
        self.nodes_dict = nodes_dict
        return nodes_dict

    def get_corner_points(self):
        """ 
        Get the 4 upper corner points of the HexagonalDomain (from left to right),
        since it's symmetrical about the middle line, we can extrapolate to obtaining 8 corner points (with 2 points duplicated)
        """
        x1 = (0-self.n)*self.delta
        x2 = (x1 + self.n*self.delta/2)
        x4 = (0+self.n)*self.delta
        x3 = (x4 - self.n*self.delta/2)
        y1 = y4 = 0
        y2 = y3 = 0 + self.n*self.eps
        x_list = [x1,x2,x3,x4]
        y_list = [y1,y2,y3,y4]

        corner_pts = []
        for x,y in zip(x_list,y_list):
            corner_pts.append(np.array([x,y,0]))
            corner_pts.append(np.array([x,-y,0]))

        return corner_pts

class RayTracing:
    """
    computes
    """
    def __init__(self,p_prime,xi_prime,HD,P_prime=1):
        """
        p_h (np.array): point where projected light ray enters the hexagonal domain
        p_prime (np.array): point where light ray enters the hexagonal domain
        xi_h (np.array): a unit vector of projected ray path 
        xi_prime (np.array): a vector of ray path
        t (np.array): target point where ray path is directed towards the hexagonal domain
        HD (class HexagonalDomain): class that contains attributes of the surface wave facet
        P_prime (int): unit radiant flux of 1
        """
        self.i = rtc.i#np.array([1,0,0])
        self.j = rtc.j#np.array([0,1,0])
        self.k = rtc.k#np.array([0,0,1])
        self.delta = rtc.delta
        self.eps = rtc.eps
        self.gamma = rtc.gamma

        self.p_h = np.array([p_prime[0],p_prime[1],0])
        self.p_prime = p_prime
        self.xi_prime = xi_prime
        self.xi_h = np.array([self.xi_prime[0],self.xi_prime[1],0])/np.linalg.norm(np.array([self.xi_prime[0],self.xi_prime[1],0]))
        self.t = self.get_target()
        self.HD = HD
        self.P_prime = P_prime

        self.r_11 = self.delta/(2*self.gamma)
        self.r_21 = self.eps/self.gamma
        self.r_12 = -1*self.r_11
        self.r_22 = self.r_21*1
        self.r_1 = np.array([self.r_11,self.r_21,0])
        self.r_2 = np.array([self.r_12,self.r_22,0])
        self.r_1_hat = np.array([-self.r_21,self.r_11,0])
        self.r_2_hat = np.array([-self.r_22,self.r_12,0])

    def get_target(self):
        v = np.array([[rtc.r_1[0],rtc.r_2[0],-self.xi_prime[0]],
                    [rtc.r_1[1],rtc.r_2[1],-self.xi_prime[1]],
                    [rtc.r_1[2],rtc.r_2[2],-self.xi_prime[2]]])
        solve_plane_vector_itxn = np.dot(np.linalg.inv(v),self.p_prime)
        t = self.p_prime + solve_plane_vector_itxn[2]*self.xi_prime
        return t

    def get_TIPs(self):
        s_k = lambda c,r_k_hat: (np.dot(2*c*self.eps*self.j, r_k_hat) - np.dot(self.p_h,r_k_hat))/np.dot(self.xi_h,r_k_hat)
        s_0 = lambda c: (c*self.eps - np.dot(self.p_h,self.j))/np.dot(self.xi_h, self.j)
        s_j_dict = {k: [] for k in range(0,3)} #where keys are the k family and values are the associated s_k
        
        for c in range(-self.HD.n,self.HD.n +1):
            for k in range(0,3):
                if k == 1:
                    r_k_hat = self.r_1_hat
                    s_j_dict[k].append(s_k(c,r_k_hat))
                elif k == 2:
                    r_k_hat = self.r_2_hat
                    s_j_dict[k].append(s_k(c,r_k_hat))
                else:
                    s_j_dict[k].append(s_0(c))
        
        #only values between 0 and s_min (minimum of the positive values) are kept as they are within the hexagon
        # print(s_j_dict)
        for k,v in s_j_dict.items():
            s = [i for i in v if i>=0]
            if len(s) > 0:
                s = s[:np.argmin(s)+1]
            else:
                s = []
            s_j_dict[k] = s

        # print(s_j_dict)
        # s_j_dict = {k:[i for i in s_j_dict[k] if i>=0] for k in s_j_dict.keys()} #filter s values >=0
        # s_j_dict = {k:s_j_dict[k][:np.argmin(s_j_dict[k])+1] for k in s_j_dict.keys()} # values are sorted in descending manner, select values until the minimum positive value
        ordered_TIPs = sorted([(k,i) for k,v in s_j_dict.items() for i in v],key=lambda x: x[1]) #list of tuples, where first element represents the k family, and 2nd element represents the associated s_j value
        ordered_TIPs = [TIP(k,index,s,self.p_h,self.xi_h) for index,(k,s) in enumerate(ordered_TIPs)]
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
        """
        nodes_list = []
        for triplets in intersect_vertices_list:
            nodes = []
            for a in triplets:
                i = (a[0]/self.delta)
                j = (a[1]/self.eps)
                k = '{:.1f},{:.1f}'.format(i,j)
                if k in self.HD.vertices_dict.keys(): #to ensure that we only consider vertices within the hexagonal domain
                    index = self.HD.vertices_dict[k]
                    v = self.HD.nodes_dict[index]
                    # n = self.HD.n_dist[index]
                    # v = a + np.array([0,0,n])
                    nodes.append(v)
            if len(nodes) == 3: #remove vertices tha are outside the hexagonal domain
                nodes_list.append(nodes)

        return nodes_list
    

    def get_normal_facet(self,intersect_vertices_list):
        """
        intersect_vertices_list (list of tuple of np.arrays): Triad Vertices that intersect with ray path
        """
        # v_list (list): nodes of v_1,v_2,v_3 (in order)
        nodes_list = self.determine_nodes(intersect_vertices_list)
        
        facets = []
        for v_list in nodes_list:
            WF = WaveFacet(v_list)
            facets.append(WF)
        
        return facets

    def get_intercepted_facets(self,WaveFacet_list,ordered_TIPs):
        """
        given the list of projected facets where TIPs lie on, get the facet where xi_prime intercepts with the wave facet
        WaveFacet_list (list of WaveFacet class): contains the attributes - nodes, norm (arranged in order of ordered_TIPs)
        ordered_TIPs (list of TIP class): TIPs ordered in order of s(j) from p_h along direction xi_h
        """
        s_q = lambda v_1,norm: np.dot((v_1 - self.p_prime),norm)/np.dot(self.xi_prime,norm)
        s_q_list = []
        for WF in WaveFacet_list:
            s = s_q(WF.nodes[0], WF.norm) #where WF.nodes[0] == v_1
            s_min = np.dot(self.xi_prime,self.xi_h)
            s_q_list.append(s*s_min)

        normal_facets_index = []
        for i in range(len(s_q_list)):
            if (s_q_list[i] > ordered_TIPs[i].s) and (s_q_list[i] < ordered_TIPs[i+1].s):
                normal_facets_index.append(i)
        
        intercepted_facets = [WaveFacet_list[i] for i in normal_facets_index]
        for wf in intercepted_facets:
            wf.target = self.get_intercepted_points(wf)

        return intercepted_facets

    def get_intercepted_points(self,WF):
        """
        returns the point where xi_prime hits the wave facet surface
        WF (a WaveFacet class) where WF belongs to the list of intercepted_facets
        """
        v_1 = WF.nodes[2] - WF.nodes[0]
        v_2 = WF.nodes[1] - WF.nodes[0]
        v = np.array([[v_1[0],v_2[0],-self.xi_prime[0]],
                    [v_1[1],v_2[1],-self.xi_prime[1]],
                    [v_1[2],v_2[2],-self.xi_prime[2]]])
        solve_plane_vector_itxn = np.dot(np.linalg.inv(v),(self.p_prime-WF.nodes[0]))
        t = self.p_prime + solve_plane_vector_itxn[2]*self.xi_prime
        return t

    def get_daughter_ray(self, WF):
        """
        WF (a WaveFacet class) where WF belongs to the list of intercepted_facets
        if angle of incidence exceeds critical angle, Total Internal Reflection (TIR) will occur --> transmittance = 0
        returns a DaughterRay class
        """
        m = 4/3 #water index of refraction
        critical_angle = np.arcsin(1/m) #critical angle for light leaving from denser medium to less dense medium
        if np.dot(self.xi_prime, WF.norm) < 0:
            # air-incident case
            xi_r = self.xi_prime - 2*np.dot(self.xi_prime,WF.norm)*WF.norm
            c = np.dot(WF.norm,self.xi_prime) - (np.dot(self.xi_prime,WF.norm)**2 + m**2 -1)**0.5
            # xi_t = (self.xi_prime - c*WF.norm)/m
            xi_t = (self.xi_prime + c*WF.norm)/m
            theta_prime = np.arccos(abs(np.dot(self.xi_prime, WF.norm))) #equiv to theta_r
            theta_t = np.arcsin(np.sin(theta_prime)/m)

            return DaughterRay(self.p_prime,self.xi_prime,theta_prime,theta_t, xi_r,xi_t,WF)
        else:
            #water incident case
            xi_r = self.xi_prime - 2*np.dot(self.xi_prime,WF.norm)*WF.norm
            theta_prime = np.arccos(abs(np.dot(self.xi_prime,WF.norm))) #equiv to theta_r
            if theta_prime > critical_angle: #if angle of incidence exceeds critical angle, Total Internal Reflection (TIR) will occur --> transmittance = 0
                return DaughterRay(self.p_prime,self.xi_prime,theta_prime, None, xi_r,None,WF)
            else:
                theta_t = np.arcsin(m*np.sin(theta_prime))
                c = np.dot(m*self.xi_prime,WF.norm) - (np.dot(m*self.xi_prime,WF.norm)**2 - m**2 + 1)**0.5
                # xi_t = m*self.xi_prime - c*WF.norm
                xi_t = m*self.xi_prime + c*WF.norm

                return DaughterRay(self.p_prime,self.xi_prime,theta_prime,theta_t, xi_r,xi_t,WF)

    def main(self):
        """
        Computes the workflow:
        1. get TIPs
        2. check if ray intercepts with any wave facets
        3. get a list of daughter rays
        """
        DR_list = []
        ordered_TIPs = self.get_TIPs()
        if len(ordered_TIPs) > 0: #check if TIPs are within the HD
            intersect_vertices_list = []
            for i in range(1,len(ordered_TIPs)):
                v = self.TIP_vertices(ordered_TIPs[i],ordered_TIPs[i-1])
                intersect_vertices_list.append(v)

            facets = self.get_normal_facet(intersect_vertices_list) #list of projected facets where TIPs lie on
            intercepted_facets = self.get_intercepted_facets(facets,ordered_TIPs)
            if len(intercepted_facets) > 0:
                for v in intercepted_facets:
                    DR = self.get_daughter_ray(v)
                    DR_list.append(DR)

        return DR_list
        

def recursive_RayTracing(stack,store_list,HD):
    """
    returns a list of typle: (idx, DaughterRay class),
        where the idx represents the series of path that a parent ray takes until the ray is traced up until that point.
        e.g. 0_r_t means that the parent ray is 0, and its reflected (r) daughter ray and the subsequent transmitted (t) daughter ray gave rise to the ray up until that point
        Note that only rays which hits a ray facet are considered, if they dont hit a ray facet, then they dont produce subsequent rays and will not have an entry
    """
    if len(stack) == 0:
        # print('store_list len'.format(len(store_list)))
        # print("End recursion")
        return store_list
    
    else:
        # print(len(stack))
        # print("Enter recursion")
        idx, s_pop = stack.pop() #DR
        RT = RayTracing(s_pop.WF.target,s_pop.xi_r,HD)
        dr_list = RT.main()
        daughter_idx =  idx + '_r' # reflected ray is postfixed with r
        # print('dr_list reflected:{}'.format(len(dr_list)))
        if len(dr_list) > 0: #push into s if dr_list is not empty
            # print('push reflected rays')
            for dr in dr_list:
                store_list.append((daughter_idx,dr)) #store daughter indx
                stack.append((daughter_idx,dr))
        
        if s_pop.xi_t is not None:
            RT = RayTracing(s_pop.WF.target,s_pop.xi_t,HD)
            dr_list = RT.main()
            daughter_idx =  idx + '_t' # transmitted ray is postfixed with r
            # print('dr_list refracted:{}'.format(len(dr_list)))
            if len(dr_list) > 0: #push into s if dr_list is not empty
                # print('push refracted rays')
                for dr in dr_list:
                    store_list.append((daughter_idx,dr))
                    stack.append((daughter_idx,dr))
        # print('Length of list: {}'.format(len(store_list)))
        # print(store_list)
        return recursive_RayTracing(stack,store_list,HD)

def RayTrace_timeseries(solar_altitude,solar_azimuth,wind_speed,save_fp,n=7,iter=5000):
    xi_prime = get_xi_prime(solar_altitude,solar_azimuth)
    np.random.seed(seed=4)
    t = np.array([rtc.delta/2,rtc.eps/2,0]) # near the middle of the hexagonal domain, did not put it in the origin as it sits at the intersection of triads
    p_prime = t-xi_prime
    dr_over_HD = {i:None for i in range(iter)}
    for i in tqdm(range(iter),desc="Rays to trace:"):
        HD = HexagonalDomain(n)
        x,y = HD.x, HD.y
        sigma = rtc.delta*(rtc.a_u/2*wind_speed)**0.5
        n_dist = np.random.normal(loc=0,scale = sigma,size = x.shape[0])
        HD.get_nodes(n_dist)
        
        # multiple_scattering = []
        # initialise first incident ray
        
        S = deque()
        L = []

        RT = RayTracing(p_prime,xi_prime,HD)

        dr_list = RT.main()
        if len(dr_list) > 0: #push into s if dr_list is not empty
            for j,dr in enumerate(dr_list):
                S.append((str(j),dr)) # seed parent rays and store their index
                L.append((str(j),dr))

        # recursive ray tracing
        all_daughter_rays = recursive_RayTracing(S,L,HD) #list of DRs from multiple scattering
        # print('{}. Number of daughter rays: {}'.format(index,len(all_daughter_rays)))

        dr_over_HD[i] = all_daughter_rays

    
    # Save results
    prefix = "solaralt{}_solarazi{}_windspeed{}".format(str(solar_altitude).zfill(2),
                                                                str(solar_azimuth).zfill(3),
                                                                str(wind_speed).zfill(2))
    
    if save_fp is not None: 
        save_daughter_rays(dr_over_HD,save_fp = save_fp,prefix=prefix)
        return dr_over_HD#organise_daughter_rays(dr_over_HD)
    else:
        organise_daughter_rays(dr_over_HD)



def RayTrace(solar_altitude,solar_azimuth,wind_speed,save_fp,n=7,step=0.3):
    """
    solar_altitude (float): in degrees, angle between horizontal surface to location of "sun"
    solar_azimuth (float): in degrees, angle between location of sun and east (+ve i direction)
    wind_speed (float): wind speed (in m/s) measured at an anemometer height of 12.5 m above mean sea level
    save_fp (str): filepath to folder
    prefix (str): prefix appended to file name
    n (int): order of HexagonalDomain
    step (float): step (float): subdivisions to divide the slanted_length/horizontal_length
    """
    HD = HexagonalDomain(n)
    x,y = HD.x, HD.y
    np.random.seed(seed=4)
    sigma = rtc.delta*(rtc.a_u/2*wind_speed)**0.5
    n_dist = np.random.normal(loc=0,scale = sigma,size = x.shape[0])
    HD.get_nodes(n_dist) # will add the attributes n_dist and nodes_dict to class
    # triang = mpl.tri.Triangulation(x, y) ## Triangulate parameter space to determine the triangles using a Delaunay triangulation

    # fig, axes = plt.subplots(subplot_kw={'projection': '3d'},figsize=(8,30))
    # axes.view_init(elev=90, azim=270)
    # axes.set_xlabel('x')
    # axes.set_ylabel('y')
    # axes.set_zlabel('z')
    # axes.plot_trisurf(triang,n_dist, linewidth=0.2, antialiased=True,cmap=plt.cm.Spectral,alpha=0.5) #3d surface
    # plt.show()

    BP = BoundaryPoints(HD,solar_altitude,solar_azimuth, step)
    xi_prime = BP.get_xi_prime()
    points_within_HD = BP.get_points_within_HD()

    dr_over_HD = {i:None for i in range(len(points_within_HD))}
    # multiple_scattering = []
    for index, p_prime in tqdm(enumerate(points_within_HD),desc="Rays to trace:"):
        # initialise first incident ray
        
        S = deque()
        L = []

        RT = RayTracing(p_prime,xi_prime,HD)

        dr_list = RT.main()
        if len(dr_list) > 0: #push into s if dr_list is not empty
            for j,dr in enumerate(dr_list):
                S.append((str(j),dr)) # seed parent rays and store their index
                L.append((str(j),dr))

        # recursive ray tracing
        all_daughter_rays = recursive_RayTracing(S,L,HD) #list of DRs from multiple scattering
        # print('{}. Number of daughter rays: {}'.format(index,len(all_daughter_rays)))

        dr_over_HD[index] = all_daughter_rays

    # Save results
    prefix = "solaralt{}_solarazi{}_windspeed{}".format(str(solar_altitude).zfill(2),
                                                                str(solar_azimuth).zfill(3),
                                                                str(wind_speed).zfill(2))
    if save_fp is not None: 
        save_daughter_rays(dr_over_HD,save_fp = save_fp,prefix=prefix)
    return dr_over_HD

def organise_daughter_rays(daughter_rays):
    for index,dr_list in daughter_rays.items(): #where keys are the index of xi_prime rays, values are list of tuple: (idx, DaughterRay class)
        # DR_dict = {i: {'DR': dict(),'WF': dict()} for i,_ in enumerate(dr_list)}#{0:{'DR':None,'WF':None},1:{'DR':None,'WF':None}}
        DR_dict = {i: {'DR': dict(),'WF': dict()} for i,_ in dr_list}

        # for i,dr in enumerate(dr_list):
        for i,dr in dr_list:
            for k,v in vars(dr).items():
                if isinstance(v,float) or isinstance(v,np.ndarray):
                    DR_dict[i]['DR'][k] = v
            for k,v in vars(dr.WF).items():
                DR_dict[i]['WF'][k] = v

        daughter_rays[index] = DR_dict

    return daughter_rays

def save_daughter_rays(daughter_rays,save_fp,prefix):
    """
    save daughter rays into json file
    daughter_rays (dict): where keys are the index of xi_prime rays, values are list of DaughterRay class
    save_fp (str): filepath to folder
    prefix (str): prefix appended to file name
    """
    if exists(save_fp) is False:
        mkdir(save_fp)

    for index,dr_list in daughter_rays.items(): #where keys are the index of xi_prime rays, values are list of DaughterRay class
        # DR_dict = {i: {'DR': dict(),'WF': dict()} for i,_ in enumerate(dr_list)}#{0:{'DR':None,'WF':None},1:{'DR':None,'WF':None}}
        DR_dict = {i: {'DR': dict(),'WF': dict()} for i,_ in dr_list}

        # for i,dr in enumerate(dr_list):
        for i,dr in dr_list:
            for k,v in vars(dr).items():
                if type(v) == np.ndarray:
                    DR_dict[i]['DR'][k] = v.tolist()
                elif isinstance(v,float) or isinstance(v,int):
                    DR_dict[i]['DR'][k] = v
            for k,v in vars(dr.WF).items():
                if type(v) == np.ndarray:
                    DR_dict[i]['WF'][k] = v.tolist()
                elif isinstance(v,float) or isinstance(v,int):
                    DR_dict[i]['WF'][k] = v
                else:
                    DR_dict[i]['WF'][k] = [i.tolist() for i in v]

        daughter_rays[index] = DR_dict

    with open(join(save_fp,"{}_DaughterRays.json".format(prefix)), 'w') as fp:
        json.dump(daughter_rays, fp)

    return

def load_daughter_rays(save_fp):
    """
    load daughter rays into a list of dictionaries
    returns a list of dictionaries
    save_fp (str): filepath to folder which contains all DR
    keys are:
        number index (int): in ascending order (from a unique ray)
            number index (int): in ascending order (indices from multiple scattering)
                DR (str): contains attributes of daughter rays
                    theta_r (float): angle of reflectance
                    theta_t (float): angle of refraction
                    xi_prime (list): unit vector of incidence ray
                    xi_r (list): unit vector of reflectance ray
                    xi_t (list): unit vector of transmitted ray
                    fresnel_reflectance (float): reflectance ratio (0 to 1)
                    fresnel_transmittance (float): transmittance ratio (0 to 1)
                WF (str): contains attributes of the wave facet the daughter ray intercepts with
                    nodes (list of list): nodes of a wave facet (3 vertices)
                    norm (list): unit normal vector of a wave facet's normal
                    target (list): point on the horizontal surface which the ray strikes
    """
    data_list = dict()
    save_fp_list = [i for i in sorted(listdir(save_fp)) if i.endswith(".json")]
    print(['{}:{}'.format(i,f.replace('_DaughterRays.json','')) for i,f in enumerate(save_fp_list)])
    for fp in save_fp_list:
        key_name = fp.replace('_DaughterRays.json','')
        with open(join(save_fp,fp), 'r') as fp:
            data = json.load(fp)

        for i,v in data.items(): #where i is the xi_prime index, v is a dict
            for j in v.keys(): # where j is the daughter ray index from multiple ray scattering
                for dr_keys,dr_values in data[i][j]['DR'].items():
                    if isinstance(dr_values,list):
                        data[i][j]['DR'][dr_keys] = np.array(data[i][j]['DR'][dr_keys])
                
                # data[i][j]['DR']['xi_r'] = np.array(data[i][j]['DR']['xi_r'])
                # data[i][j]['DR']['xi_t'] = np.array(data[i][j]['DR']['xi_t'])
                # data[i][j]['DR']['xi_prime'] = np.array(data[i][j]['DR']['xi_prime'])
                data[i][j]['WF']['nodes'] = [np.array(i) for i in data[i][j]['WF']['nodes']]
                data[i][j]['WF']['norm'] = np.array(data[i][j]['WF']['norm'])
                data[i][j]['WF']['target'] = np.array(data[i][j]['WF']['target'])
        
        # data_copy = {int(i):{int(j):v_2 for j,v_2 in v_1.items()} for i,v_1 in data.items()}
        data_copy = {int(i):{j:v_2 for j,v_2 in v_1.items()} for i,v_1 in data.items()}
        data_list[key_name] = data_copy

    return data_list

class IrradianceReflectance:
    """
    daughter_rays (dict): keys are:
        indices (str): if indices are digits, they are parent rays from light source, else, they are daughter rays/descendants
            DR (str)
                p_prime
                xi_prime: parent ray
                xi_r: daughter ray
                xi_t: daughter ray
                theta_r
                theta_t
                fresnel_reflectance
                fresnel_transmittance
            WF (str): wave facet
                nodes
                norm
                tilt
                target: location where xi_prime intersects with wave facet to produce xi_r and xi_t
    P_prime (int): unit radiant flux of 1
    alpha (float) (0-inf): volume attenuation coefficient for the water to account for flux lost from the ray during water transmission. Setting alpha=0 in effect makes the water transparent. 
        Setting alpha = inf eliminates any multiple scattering effects arising from subsurface travels of daughter rays
    ======================================
    TEST CASES
    d = {0:{
        '0':{'DR':{'xi_r':np.array([0,0,-1]),'fresnel_reflectance':0.5},'radiance':1},
        '0_r':{'DR':{'xi_r':np.array([0,0,1]),'fresnel_reflectance':0.5,'xi_t':np.array([0,0,1]),'fresnel_transmittance':0.5},'radiance':0.5},
        '0_r_r':{'DR':{'xi_r':np.array([0,0,-1]),'fresnel_reflectance':0.5,'xi_t':np.array([0,0,1]),'fresnel_transmittance':0.5},'radiance':0.25},
        '0_r_r_r':{'DR':{'xi_r':np.array([0,0,-1]),'xi_t':np.array([0,0,1]),'fresnel_reflectance':0.5},'radiance':0.125},
        '0_r_r_r_t':{'DR':{'xi_r':np.array([0,0,1]),'fresnel_reflectance':0.5},'radiance':0.0625},

        '1':{'DR':{'xi_r':np.array([0,0,-1]),'fresnel_reflectance':0.5},'radiance':1},
        '1_r':{'DR':{'xi_t':np.array([0,0,1]),'fresnel_reflectance':0.5,'xi_r':np.array([0,0,1]),'fresnel_transmittance':0.5},'radiance':0.5}
        },
    }

    d = {0:{
        '0':{'DR':{'xi_r':np.array([0,0,1]),'fresnel_reflectance':1},'radiance':1},
        '1':{'DR':{'xi_r':np.array([0,0,1]),'fresnel_reflectance':1},'radiance':1},
        '2':{'DR':{'xi_r':np.array([0,0,1]),'fresnel_reflectance':1},'radiance':1}
        },
    }
    =======================================
    >>> IR = IrradianceReflectance(data_list)
    >>> IR.albedo() -> returns a dict of the reflectance for each parent ray (random sea surface realisation)
    """
    def __init__(self, data_list, P_prime=1, alpha=0):
        self.data_list = data_list
        self.P_prime = P_prime
        self.alpha = alpha
    
    def compute_radiance(self,daughter_rays):
        
        daughter_rays_copy = daughter_rays.copy()

        def recursive_radiance(idx):
            """
            recursion with memoisation. Since memoisation overwrites the results, to run with new analysis, one will need to reload the original daughter_rays
            """
            if 'radiance' in daughter_rays_copy[idx].keys():
                # end recursion condition
                return daughter_rays_copy[idx]['radiance'] # memoisation to avoid computing repeatedly for other rays
            elif idx.isdigit():
                # end recursion condition
                daughter_rays_copy[idx]['radiance'] = self.P_prime
                return self.P_prime
            else:
                idx_parent = idx[:-2]
                xi_prime = daughter_rays[idx_parent]['DR']['xi_prime']
                WF_norm = daughter_rays[idx_parent]['WF']['norm']
                if np.dot(xi_prime, WF_norm) > 0:
                    t_parent = daughter_rays[idx_parent]['WF']['target']
                    t_daughter = daughter_rays[idx]['WF']['target']
                    d = np.linalg.norm(t_parent - t_daughter)
                    exponential_factor = np.exp(-d*self.alpha)
                else:
                    exponential_factor = None

                if idx.endswith('_t'):
                    if 'fresnel_transmittance' not in daughter_rays[idx_parent]['DR'].keys():
                        daughter_rays[idx_parent]['DR']['fresnel_transmittance'] = 0
                    parent_radiance = daughter_rays[idx_parent]['DR']['fresnel_transmittance']
                    
                else:
                    idx_parent = idx[:-2]
                    if 'fresnel_reflectance' not in daughter_rays[idx_parent]['DR'].keys():
                        daughter_rays[idx_parent]['DR']['fresnel_reflectance'] = 1
                    parent_radiance = daughter_rays[idx_parent]['DR']['fresnel_reflectance']
                    
                parent_radiance = parent_radiance if exponential_factor is None else parent_radiance*exponential_factor
                return parent_radiance*recursive_radiance(idx_parent)
        
        for k in daughter_rays.keys():
            daughter_rays_copy[k]['radiance'] = recursive_radiance(k)

        return daughter_rays_copy

    def recursive_upwelling_radiance(self,daughter_rays):
        
        def tally_radiance(parent_node,d,radiance_list):
            if parent_node.reflected is None and parent_node.transmitted is None:
                if d[parent_node.key]['DR']['xi_r'][2] > 0:
                    radiance = d[parent_node.key]['radiance']
                    fresnel = d[parent_node.key]['DR']['fresnel_reflectance']
                    radiance_list.append(radiance*fresnel)

                if 'xi_t' in d[parent_node.key]['DR'].keys():
                    if d[parent_node.key]['DR']['xi_t'][2] > 0:
                        radiance = d[parent_node.key]['radiance']
                        fresnel = d[parent_node.key]['DR']['fresnel_transmittance']
                        radiance_list.append(radiance*fresnel)
                        
                return radiance_list
            else:
                if parent_node.reflected is not None:
                    tally_radiance(parent_node.reflected,d,radiance_list)

                else:
                    if d[parent_node.key]['DR']['xi_r'][2] > 0:
                        radiance = d[parent_node.key]['radiance']
                        fresnel = d[parent_node.key]['DR']['fresnel_reflectance']
                        radiance_list.append(radiance*fresnel)
                
                if parent_node.transmitted is not None:
                    tally_radiance(parent_node.transmitted,d,radiance_list)
                else:
                    if 'xi_t' in d[parent_node.key]['DR'].keys():
                        if d[parent_node.key]['DR']['xi_t'][2] > 0:
                            radiance = d[parent_node.key]['radiance']
                            fresnel = d[parent_node.key]['DR']['fresnel_transmittance']
                            radiance_list.append(radiance*fresnel)
                
                return radiance_list

        
        F = Forest()
        [F.grow_trees(k) for k in daughter_rays.keys()]
        # F.print_nodes()
        radiance = []
        for k in F.tree_list.keys():
            radiance_list = []
            r = tally_radiance(F.tree_list[k],daughter_rays,radiance_list)
            radiance.append(r)
        
        return radiance

    def albedo(self):
        """
        albedo or irradiance reflectance r_(xi')
        returns the ensemble average over all realisations of S(omega)
        """
        albedo_dict = dict()

        for k,v in self.data_list.items():
            if bool(v) is True:
                v = self.compute_radiance(v)
                L = self.recursive_upwelling_radiance(v)
                albedo_dict[k] = np.nansum([np.nansum(l) for l in L])/(len(L)*self.P_prime)

        return albedo_dict

def plot_irradiance_reflectance(data_list,P_prime=1):
    """
    returns a graph of irradiance reflectance as a function of wind_speed, solar altitude and azimuth
    """
    tally_irradiance = {int(parse_solar_attributes(k)[0]): dict() for k in data_list.keys()}

    color_step = np.linspace(0,1,len(tally_irradiance.keys()))
    for k in data_list.keys():
        alt,azi,U = parse_solar_attributes(k)
        tally_irradiance[int(alt)][int(azi)] = dict()

    for k in data_list.keys():
        alt,azi,U = parse_solar_attributes(k)
        tally_irradiance[int(alt)][int(azi)][int(U)] = k

    
    plt.figure(figsize=(10,8))
    for i, (alt, d) in zip(color_step,tally_irradiance.items()):
        for azi, d1 in d.items():
            albedo_U = [] # albedo as a function of windspeed
            U_list = []
            for U, key in d1.items():
                ir = IrradianceReflectance(data_list[key])
                albedo = ir.albedo()
                avg_albedo = np.nansum([v for v in albedo.values()])/(len(albedo.keys())*P_prime)
                albedo_U.append(avg_albedo)
                U_list.append(U)
            
            if azi == 90:
                label = f'crosswind, alt: {alt}'
                plt.plot(U_list,albedo_U, linestyle='--',color=plt.cm.viridis(i),label=label)
            else:
                label = f'alongwind, alt: {alt}'
                plt.plot(U_list,albedo_U,color=plt.cm.viridis(i),label=label)
    plt.xlabel('Windspeed (m/s)')
    plt.ylabel('Irradiance reflectance ratio')
    plt.yscale("log")
    plt.tick_params(axis='y', which='minor')
    ax = plt.gca()
    ax.yaxis.set_minor_formatter(FormatStrFormatter("%.3f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    plt.legend(loc='center left', bbox_to_anchor=(1,0.5))
    plt.show()

def parse_solar_attributes(key_name):
    """
    key_name (str): key name of data list (loaded from load_daughter_rays)
    """
    alt,azi,U = key_name.split('_')
    alt = float(alt.replace('solaralt',''))
    azi = float(azi.replace('solarazi',''))
    U = float(U.replace('windspeed',''))
    return (alt,azi,U)


def plot_daughter_rays(data_list,elev=90,azim = 270):
    """
    data_list (dict): dictionary that contains multiple scattering of daughter rays
    """
    fig, axes = plt.subplots(subplot_kw={'projection': '3d'},figsize=(10,10))
    legend_dict = {'reflected':'green','refracted':'orange','norm':'blue','incident':'red'}
    for k,d in data_list.items():
        nodes = d['WF']['nodes']
        target = d['WF']['target']
        norm = d['WF']['norm']
        xi_r = d['DR']['xi_r']
        try:
            xi_t = d['DR']['xi_t']
        except:
            xi_t = None
        xi_prime = d['DR']['xi_prime']
        
        axes.arrow3D(target[0],target[1],target[2],
            dx = xi_r[0], dy = xi_r[1], dz = xi_r[2],
            mutation_scale=10,fc='green') #reflected vector
        if xi_t is not None:
            axes.arrow3D(target[0],target[1],target[2],
                        dx = xi_t[0], dy = xi_t[1], dz = xi_t[2],
                        mutation_scale=10,fc='orange') #refracted vector
        axes.arrow3D(target[0],target[1],target[2],
                    dx = norm[0], dy = norm[1], dz = norm[2],
                    mutation_scale=10,fc='blue') #norm
        axes.arrow3D(target[0],target[1],target[2],
                    dx = xi_prime[0],
                    dy = xi_prime[1],
                    dz = xi_prime[2],mutation_scale=10,fc = 'red') #incident vector
        axes.plot_trisurf([i[0] for i in nodes],
            [i[1] for i in nodes],
            [i[2] for i in nodes],alpha=0.7)
        axes.text(target[0],target[1],target[2],"{}".format(k))
    
    axes.set_xlabel('x')
    axes.set_ylabel('y')
    axes.set_zlabel('z')
    axes.set_proj_type('ortho')
    axes.view_init(elev=elev, azim=azim)
    patchList = []
    for key in legend_dict:
        data_key = mpatches.Patch(color=legend_dict[key], label=key)
        patchList.append(data_key)

    plt.legend(handles=patchList)
    plt.show()
    return

def multiple_scattering(data_list,thresh=1):
    """
    data_list (dict): dictionary that contains multiple scattering of daughter rays
    thresh (int): threshold of multiple scattering
    returns % of ray tracing that goes through multiple scattering to produce > thresh daughter rays
    """
    ms = dict()
    for k,l in data_list.items():
        ms[k] = len([k for k,v in l.items() if len(v) > thresh])/len(l.keys())*100
    return ms

def plot_multiple_scattering(data_list):
    """
    plot multiple scattering as a function of wind speed, solar azimuth and altitude
    """
    tally_ms = {int(parse_solar_attributes(k)[0]): dict() for k in data_list.keys()}

    color_step = np.linspace(0,1,len(tally_ms.keys()))
    for k in data_list.keys():
        alt,azi,U = parse_solar_attributes(k)
        tally_ms[int(alt)][int(azi)] = dict()

    for k in data_list.keys():
        alt,azi,U = parse_solar_attributes(k)
        tally_ms[int(alt)][int(azi)][int(U)] = k


    plt.figure(figsize=(10,8))
    for i, (alt, d) in zip(color_step,tally_ms.items()):
        for azi, d1 in d.items():
            ms_U = [] # multiple scattering as a function of windspeed
            U_list = []
            for U, key in d1.items():
                U_list.append(U)
                ms = dict()
                for k,v in data_list[key].items():
                    ms[k] = sum([False if ray_index.isdigit() else True for ray_index in v.keys()])
                    
                perc_ms = sum([v for v in ms.values()])/len(ms.keys())
                ms_U.append(perc_ms)
            if azi == 90:
                label = f'crosswind, alt: {alt}'
                plt.plot(U_list,ms_U, linestyle='--',color=plt.cm.viridis(i),label=label,linewidth=2)
            else:
                label = f'alongwind, alt: {alt}'
                plt.plot(U_list,ms_U,color=plt.cm.viridis(i),label=label,linewidth=2)

    plt.xlabel('Windspeed (m/s)')
    plt.ylabel('Average multiple scattering')
    plt.yscale("log")
    plt.tick_params(axis='y', which='minor')
    ax = plt.gca()
    ax.yaxis.set_minor_formatter(FormatStrFormatter("%.4f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    plt.legend(loc='center left', bbox_to_anchor=(1,0.5))
    plt.show()
    return

class GlitterPattern:
    def __init__(self,data_list,xi_prime,camera_axis,wind_speed,f=1,fov_h=30,n=100,ax=None,plot=True):
        """
        data_list (dict): keys are:
            indices of xi_prime
                indices of multiple ray scattering
        f (float): 1 as default for generalisability. Simulates focal length of camera
        camera axis (a)  (np.array): is aligned along the specular ray path such that theta_s' = theta_c (angle of reflection)
        fov_h (float): horizontal viewing angle of camera
        n (int): number of points to generate the contour lines of alpha and beta
        """
        self.data_list = data_list
        self.wind_speed = wind_speed
        self.ax = ax #axis to plot the figure
        # camera position is fixed with f and a

        self.xi_prime = xi_prime/np.linalg.norm(xi_prime) # direction of the sun rays (must be fixed since sun location is fixed at an instance of wave surface realisation)

        self.f = f # distance from S along camera axis direction
        # a = xi_prime - 2*np.dot(xi_prime,self.z)*self.z # camera axis direction (a)
        self.camera_axis = camera_axis/np.linalg.norm(camera_axis)

        # sun-based coordinate system in wind-based coordinates
        self.z = rtc.k#np.array([0,0,1]) #same as wind_based
        x = np.array([self.xi_prime[0],self.xi_prime[1],0]) #x is in the same direction as xi_prime
        self.x = x/np.linalg.norm(x)
        y = np.cross(self.z,self.x)
        self.y = y/np.linalg.norm(y)

        # image plane coordinate system in wind-based coordinates
        self.h = -self.y
        v = np.cross(self.camera_axis,self.h)
        self.v = v/np.linalg.norm(v)
        self.fov_h = fov_h
        self.n = n
        self.plot = plot

        # wind-based coordinate system
        self.k = self.z
        self.i = rtc.i#np.array([1,0,0])
        self.j = rtc.j#np.array([0,1,0])

        # calculate solar angles wrt to wind-based direction
        ray_azimuth = np.arctan(self.xi_prime[1]/self.xi_prime[0])  #angle (in rad) between i and ray direction (phi)
        self.ray_zenith = np.arccos(self.xi_prime[2])

        if np.sign(self.xi_prime[1]) >= 0 and np.sign(self.xi_prime[0]) >= 0:
            self.ray_azimuth = ray_azimuth
            self.solar_azimuth = ray_azimuth + np.pi
        elif np.sign(self.xi_prime[1]) >= 0 and np.sign(self.xi_prime[0]) < 0:
            self.ray_azimuth = ray_azimuth + np.pi
            self.solar_azimuth = ray_azimuth
        elif np.sign(self.xi_prime[1]) < 0 and np.sign(self.xi_prime[0]) < 0:
            self.ray_azimuth = ray_azimuth + np.pi
            self.solar_azimuth = ray_azimuth
        else:
            self.ray_azimuth = 2*np.pi + ray_azimuth
            self.solar_azimuth = self.ray_azimuth - np.pi

        self.phi_prime = self.ray_azimuth # angle in rad #angle between x and i in rad
        
        # # location of light source
        self.solar_zenith = (np.pi - self.ray_zenith)/np.pi*180 # in deg
        self.solar_azimuth = self.solar_azimuth/np.pi*180 # in deg
        
        # camera location wrt to wind-based direction
        # self.camera_azimuth = np.arctan(self.camera_axis[1]/self.camera_axis[0])/np.pi*180 #in deg
        self.camera_zenith = np.arccos(self.camera_axis[2])/np.pi*180 #in deg
        camera_azimuth = np.arctan(self.camera_axis[1]/self.camera_axis[0])  #angle (in rad) between i and ray direction (phi)
        if np.sign(self.camera_axis[1]) >= 0 and np.sign(self.camera_axis[0]) >= 0:
            self.camera_azimuth = camera_azimuth
        elif np.sign(self.camera_axis[1]) >= 0 and np.sign(self.camera_axis[0]) < 0:
            self.camera_azimuth = camera_azimuth + np.pi
        elif np.sign(self.camera_axis[1]) < 0 and np.sign(self.camera_axis[0]) < 0:
            self.camera_azimuth = camera_azimuth + np.pi
        else:
            self.camera_azimuth = 2*np.pi + camera_azimuth
        
        self.camera_azimuth = self.camera_azimuth/np.pi*180
        # camera viewing angle in deg
        self.fov_v_upper = 90 - self.camera_zenith #viewing angle of camera that corresponds to looking toward ocean horizon
        self.fov_v_lower = -self.camera_zenith
    
    def get_psi(self,xi_r):
        """
        xi_r (np.array): wind-based reflected ray
        get glitter locations in terms of camera viewing angles in wind-based coordinate system
        returns a tuple (psi_h,psi_v) in deg
        """
        
        xi_r = xi_r/np.linalg.norm(xi_r) #change into unit vector
        wind_to_sun_mat = wind_to_sun_rot(self.phi_prime)
        xi_r = np.dot(wind_to_sun_mat,xi_r) # (xi_x,xi_y,xi_z)
        a = np.dot(wind_to_sun_mat,self.camera_axis) #(a_x,a_y,a_z)
        t_h = -self.f*xi_r[1]/np.dot(xi_r,a)
        t_v = self.f*(xi_r[0]*a[2] - xi_r[2]*a[0])/np.dot(xi_r,a)
        # t_h = self.f*np.dot(xi_r,self.h)/np.dot(xi_r,a)
        # t_v = self.f*np.dot(xi_r,self.v)/np.dot(xi_r,a)
        psi_h = np.arctan(t_h/self.f)/np.pi*180
        psi_v = np.arctan(t_v/self.f)/np.pi*180
        # contraint glittern pattern to camera viewing angles
        if ((psi_h < self.fov_h) and (psi_h > -self.fov_h)) and ((psi_v < self.fov_v_upper) and (psi_v > self.fov_v_lower)):
            return {'psi_h':psi_h,'psi_v':psi_v,'t_h':t_h,'t_v':t_v}#(psi_h,psi_v)
        else:
            return None#None, None

    def plot_glitter_pattern(self,fov_lower=None,fov_upper=None):
        """
        xi_r and a must face the same direction s.t. xi \cdot a > 0 such that specular glitter will appear on the image
        iterate across all reflected/transmitted ray and check if xi \cdot a > 0, then compute t_h and t_v
        """
        output = []
        for v1 in self.data_list.values(): # iterate across individual rays
            if bool(v1) is True: #check if dict is not empty
                for v2 in v1.values():
                    xi_r = v2['DR']['xi_r']
                    if 'xi_t' in v2['DR'].keys():
                        xi_t = v2['DR']['xi_t']
                    else:
                        xi_t = None
                    if np.dot(xi_r,self.camera_axis) > 0: # camera axis and reflected ray is in the same direction
                        psi = self.get_psi(xi_r)
                        if psi is not None:
                            # d = {'psi_h':psi_h,'psi_v':psi_v} # in degrees
                            output.append(psi)
                    if (xi_t is not None) and (np.dot(xi_t,self.camera_axis) > 0):
                        psi = self.get_psi(xi_t)
                        if psi is not None:
                            # d = {'psi_h':psi_h,'psi_v':psi_v} # in degrees
                            output.append(psi)
        if self.ax is None:
            ax = self.glitter_pattern_contour(fov_lower,fov_upper)
            ax.plot([i['psi_h'] for i in output],[i['psi_v'] for i in output],'k.')
            plt.show()
        else:
            self.glitter_pattern_contour(fov_lower,fov_upper)
            self.ax.plot([i['psi_h'] for i in output],[i['psi_v'] for i in output],'k.')
        
        return output

    def glitter_pattern_contour(self,fov_lower=None,fov_upper=None):
        """
        a (wind-based coordinate system)
        xi_prime (wind-based coordinate system)
        returns isolines (dict) where keys are alpha and beta in degree
        """

        #matrix for rotating wind based vector to sun-based vector such that we are viewing wrt to the specular direction of sun rays
        wind_to_sun_mat = wind_to_sun_rot(self.phi_prime)
        # get xi_prime and camera-axis (a) in sun-based coordinate system
        xi_prime = np.dot(wind_to_sun_mat,self.xi_prime)
        xi_prime = xi_prime/np.linalg.norm(xi_prime)
        a = np.dot(wind_to_sun_mat,self.camera_axis) #(a_x,a_y,a_z)
        a = a/np.linalg.norm(a)
        if (fov_lower is not None) and (fov_upper is not None):
            psi_h,psi_v = np.meshgrid(np.linspace(-self.fov_h,self.fov_h,self.n),np.linspace(fov_lower,fov_upper,self.n))
        else:
            psi_h,psi_v = np.meshgrid(np.linspace(-self.fov_h,self.fov_h,self.n),np.linspace(self.fov_v_lower,self.fov_v_upper,self.n))
        alpha_list = []
        beta_list = []
        for p_h,p_v in zip(psi_h.flatten(),psi_v.flatten()):
            t_h = self.f*np.tan(p_h/180*np.pi)
            t_v = self.f*np.tan(p_v/180*np.pi)
            xi_x = ((self.f**2 + t_h**2 + t_v**2)**(-0.5))*(self.f*a[0] + t_v*a[2])
            xi_y = ((self.f**2 + t_h**2 + t_v**2)**(-0.5))*(-t_h)
            xi_z = ((self.f**2 + t_h**2 + t_v**2)**(-0.5))*(self.f*a[2] - a[0]*t_v)
            xi = np.array([xi_x,xi_y,xi_z])
            n = (xi - xi_prime)
            n = n/np.linalg.norm(n) #wrt to sun-based coordinate
            alpha = np.arctan(-n[1]/n[0])/np.pi*180 #azimuth angle wrt sun-based coordinate system, positive clockwise from i
            # if p_v <= 0:
            #     alpha = alpha + 180
            # elif p_v > 0 and p_h <= 0:
            #     alpha = 360 + alpha
            beta = np.arccos(n[2])/np.pi*180 #tilt angle wrt sun-based coordinate system
            alpha_list.append(alpha)
            beta_list.append(beta)
        
        alpha = np.array(alpha_list).reshape(psi_h.shape) # azimuth angle of wave facet, positive clockwise from i
        beta = np.array(beta_list).reshape(psi_h.shape) # tilt angle of wave facet

        if self.plot is True:
            fontsize=8

            def fmt_alpha(x):
                s = r"$\alpha={x:.0f}$".format(x=x)
                return s

            def fmt_beta(x):
                s = r"$\beta={x:.0f}$".format(x=x)
                return s

            if self.ax is not None:
                CS_alpha = self.ax.contour(psi_h,psi_v,alpha,colors='k',linestyles='dashed')
                self.ax.clabel(CS_alpha, inline=True, fontsize=fontsize,fmt=fmt_alpha)
                CS_beta = self.ax.contour(psi_h,psi_v,beta,colors='k',linestyles='dashed')
                self.ax.clabel(CS_beta, inline=True, fontsize=fontsize,fmt=fmt_beta)
                self.ax.set_title(r'$\phi_s = {azimuth:.2f}, \theta_s = {zenith:.2f}; \phi_c = {c_azi:.2f}, \theta_c = {c_theta:.2f}, U = {wind_speed:.1f} m/s$'.format(
                                                                    azimuth=self.solar_azimuth,zenith=self.solar_zenith,
                                                                    c_azi=self.camera_azimuth,c_theta=self.camera_zenith,wind_speed=self.wind_speed))
                self.ax.axes.set_aspect('equal')
                self.ax.set_xlabel(r'$\psi_h$')
                self.ax.set_ylabel(r'$\psi_v$')
            else:
                fig, ax = plt.subplots(figsize=(10,10))
                CS_alpha = ax.contour(psi_h,psi_v,alpha,colors='k',linestyles='dashed')
                ax.clabel(CS_alpha, inline=True, fontsize=fontsize,fmt=fmt_alpha)
                CS_beta = ax.contour(psi_h,psi_v,beta,colors='k',linestyles='dashed')
                ax.clabel(CS_beta, inline=True, fontsize=fontsize,fmt=fmt_beta)
                ax.set_title(r'$\phi_s = {azimuth:.2f}, \theta_s = {zenith:.2f}; \phi_c = {c_azi:.2f}, \theta_c = {c_theta:.2f}, U = {wind_speed:.1f} m/s$'.format(
                                                                    azimuth=self.solar_azimuth,zenith=self.solar_zenith,
                                                                    c_azi=self.camera_azimuth,c_theta=self.camera_zenith,wind_speed=self.wind_speed))
                ax.axes.set_aspect('equal')
                ax.set_xlabel(r'$\psi_h$')
                ax.set_ylabel(r'$\psi_v$')
                # plt.show()
                return ax

        else:
            return {'psi_h':psi_h,'psi_v':psi_v,'alpha':alpha,'beta':beta}

def wind_to_sun_rot(phi_prime):
    """
    phi_s_prime: angle (in rad) measured between i and location of sun (light source) in counterclockwise
    phi_prime: angle (in rad) measured wrt between x (in the same direction as xi_prime) and i-axis in counterclockwise
    rotation matrix to rotate wind-based vector to sun based vector
    """
    # phi_prime = phi_s_prime + np.pi
    return np.array([[np.cos(phi_prime),np.sin(phi_prime),0],
                [-np.sin(phi_prime),np.cos(phi_prime),0],
                [0,0,1]])

def sun_to_wind_rot(phi_prime):
    """
    phi_s_prime: angle (in rad) measured between i and location of sun (light source) in counterclockwise
    phi_prime: angle (in rad) measured wrt between x (in the same direction as xi_prime) and i-axis in counterclockwise
    rotation matrix to rotate wind-based vector to sun based vector
    """
    # phi_prime = phi_s_prime + np.pi
    return np.array([[np.cos(phi_prime),-np.sin(phi_prime),0],
                [np.sin(phi_prime),np.cos(phi_prime),0],
                [0,0,1]])

def get_xi_prime(solar_altitude,solar_azimuth):
    """
    returns unit xi_prime, directed downwards, towards azimuth
    """
    # location of sun
    solar_zenith = (90 - solar_altitude)
    theta_s = solar_zenith/180*np.pi
    phi_s = solar_azimuth/180*np.pi
    # direction of where ray is travelling
    theta = np.pi - theta_s
    phi = (phi_s + np.pi)%(2*np.pi) #modulo 360
    xi_prime = np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
    t = xi_prime/np.linalg.norm(xi_prime) # ray is directed
    
    return t

def get_camera_vector(camera_altitude, camera_azimuth):
    """
    returns location of camera
    """
    camera_zenith = (90 - camera_altitude)
    theta_c = camera_zenith/180*np.pi
    phi_c = camera_azimuth/180*np.pi
    xi_prime = np.array([np.sin(theta_c)*np.cos(phi_c),np.sin(theta_c)*np.sin(phi_c),np.cos(theta_c)])
    t = xi_prime/np.linalg.norm(xi_prime)
    return t

if __name__ == '__main__':
    # raytracing
    daughter_rays = RayTrace_timeseries(args.solar_altitude, 
                        args.solar_azimuth, 
                        args.wind_speed, 
                        save_fp=args.save_fp, 
                        n=args.n, 
                        iter=args.iter)
    # plot glitter
    xi_prime = get_xi_prime(args.solar_altitude,args.solar_azimuth)
    camera_axis = get_camera_vector(args.camera_altitude, args.camera_azimuth)
    GP = GlitterPattern(daughter_rays,xi_prime,camera_axis,args.wind_speed)
    GP.plot_glitter_pattern()