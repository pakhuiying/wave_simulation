import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from math import ceil
import random
import ray_tracing_constants as rtc

class BoundaryPoints:
    def __init__(self,HD, solar_altitude,step):
        """
        HD (HexagonalDomain class)
        solar_altitude (float): in degrees, angle between horizontal surface to "sun"
        num_points (int): number of subdivisions to divide the slanted_length/horizontal_length
            i.e. number of points to generate per surface boundary line
        # s (np.array): point on the vector
        # l (float): scaling factor of vector
        # r_1 is pointing towards +ve i and +ve j direction
        # r_2 is pointing towards -ve i and +ve j direction
        # i_vector is point towards +ve i
        """
        self.n = HD.n
        self.solar_altitude = solar_altitude
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
        self.slanted_division = np.arange(0,self.slanted_length+self.step,self.step)
        self.horizontal_division = np.arange(0,self.horizontal_length+self.step,self.step)
        
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
            for l in range(len(self.slanted_division)): #iterate across all the subdivisions along the length
                if i%2 == 0: #if either the r1 or r2 vector
                    points_dict[coord_start_index].append(v(self.corner_points[coord_start_index],self.slanted_division[l]))
                    points_dict[coord_start_index_rev].append(v(self.corner_points[coord_start_index_rev],-self.slanted_division[l]))
                else: #if it's the i_vector
                    points_dict[coord_start_index].append(v(self.corner_points[coord_start_index],self.horizontal_division[l]))
                    points_dict[coord_start_index_rev].append(v(self.corner_points[coord_start_index_rev],-self.horizontal_division[l]))
        
        return points_dict

    def get_points_within_HD(self):
        """
        step (float): distance between adjacent points
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
                points_list.append(self.r2_vector(s,-l))
        return points_list
    
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
        self.i = np.array([1,0,0])
        self.j = np.array([0,1,0])
        self.k = np.array([0,0,1])
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
    
    def get_norm(self):
        n1 = np.cross((self.nodes[2] - self.nodes[0]),(self.nodes[1] - self.nodes[0]))
        n1 = n1/np.linalg.norm(n1)

        if np.dot(n1,np.array([0,0,1])) < 0:
            sign = -1
        else:
            sign = 1

        n1 = sign*n1
        return n1

class DaughterRay:
    """
    theta_r (float): angle of incidence = reflectance (in rad)
    theta_t (float): angle of refraction (in rad)
    xi_r (np.array): vector of reflectance
    xi_t (np.array): vector of transmittance
    WF (WaveFacet class): wave facet where incident ray intercepts with WaveFacet
    returns the attributes of a daughter ray
    """
    def __init__(self,theta_r,theta_t,xi_r,xi_t,WF):
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
        if (self.theta_t is not None) and (self.xi_t is not None):
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
        self.i = np.array([1,0,0])
        self.j = np.array([0,1,0])
        self.k = np.array([0,0,1])
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
        self.i = np.array([1,0,0])
        self.j = np.array([0,1,0])
        self.k = np.array([0,0,1])
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

        s_j_dict = {k:[i for i in s_j_dict[k] if i>=0] for k in s_j_dict.keys()} #filter s values >=0
        s_j_dict = {k:s_j_dict[k][:np.argmin(s_j_dict[k])+1] for k in s_j_dict.keys()} # values are sorted in descending manner, select values until the minimum positive value
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

            return DaughterRay(theta_prime,theta_t, xi_r,xi_t,WF)
        else:
            #water incident case
            theta_prime = np.arccos(abs(np.dot(self.xi_prime,WF.norm))) #equiv to theta_r
            theta_t = np.arcsin(m*np.sin(theta_prime))
            xi_r = self.xi_prime - 2*np.dot(self.xi_prime,WF.norm)*WF.norm
            if theta_prime > critical_angle: #if angle of incidence exceeds critical angle, Total Internal Reflection (TIR) will occur --> transmittance = 0
                return DaughterRay(theta_prime, None, xi_r,None)
            else:
                
                c = np.dot(m*self.xi_prime,WF.norm) - (np.dot(m*self.xi_prime,WF.norm)**2 - m**2 + 1)**0.5
                # xi_t = m*self.xi_prime - c*WF.norm
                xi_t = m*self.xi_prime + c*WF.norm

                return DaughterRay(theta_prime,theta_t, xi_r,xi_t,WF)

    def main(self):
        """
        Computes the workflow:
        1. get TIPs
        2. check if ray intercepts with any wave facets
        3. get a list of daughter rays
        """
        DR_list = []
        ordered_TIPs = self.get_TIPs()

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
    if len(stack) == 0:
        # print('store_list len'.format(len(store_list)))
        # print("End recursion")
        return store_list
    
    else:
        print(len(stack))
        # print("Enter recursion")
        s_pop = stack.pop() #DR
        RT = RayTracing(s_pop.WF.target,s_pop.xi_r,HD)
        dr_list = RT.main()
        # print('dr_list reflected:{}'.format(len(dr_list)))
        if len(dr_list) > 0: #push into s if dr_list is not empty
            # print('push reflected rays')
            for dr in dr_list:
                store_list.append(dr)
                stack.append(dr)
        
        if s_pop.xi_t is not None:
            RT = RayTracing(s_pop.WF.target,s_pop.xi_t,HD)
            dr_list = RT.main()
            # print('dr_list refracted:{}'.format(len(dr_list)))
            if len(dr_list) > 0: #push into s if dr_list is not empty
                # print('push refracted rays')
                for dr in dr_list:
                    store_list.append(dr)
                    stack.append(dr)
        # print('Length of list: {}'.format(len(store_list)))
        # print(store_list)
        return recursive_RayTracing(stack,store_list,HD)

        