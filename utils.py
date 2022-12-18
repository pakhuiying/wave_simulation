import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

from time import perf_counter as timer

"""
Name: 2dwaavesim
author='Cooper Hatfield',
author_email='cooperhatfield@yahoo.ca',
description='Simulate waves on 2D surfaces with arbitrary shape/size!',
url='https://github.com/cooperhatfield/2dwavesim'
"""
# define constants of simulation
tf = 4 #s
c = 350.0 #m/s
g = 2e-2 #1/s
ds = 1 #m


# find the time step that satisfies CFL condition 
dt = (ds**(-2) + ds**(-2))**(-1/2) /c #s

# truncate time step to 5 decimal places to compensate 
# for any rounding errors
dt *= 1e5
dt = np.floor(dt)
dt /= 1e5


# define constants to help later
C = (c * dt / ds)**2
G = (g * dt) / 2


# define array of times
times = np.arange(0,tf,dt)


'''This is the 2D room system. It is a 3D array with    '''

class Room_2D:
    
    def Apply_Mask(self, t: int):
        pass
    
    
    def Poly_Mask(self, vertices, ds, attenuation):
        '''Function that takes in a list of points representing
        the vertices of the polygon that is the shape of the room,
        and a spacial step length. This function defines a MatPlotLib 
        Path object from the vertices given, and uses that to create
        an array of 1s and 0s to mask the solution array by elementwise
        multiplication. It returns the mask array.'''

        # define the polygon that is the room shape
        self.poly = Path(vertices)

        # find the coordinate bounds of the room
        self.points = [*zip(*vertices)]
        self.max_x = max(self.points[0]) + 1
        self.min_x = min(self.points[0])
        self.max_y = max(self.points[1]) + 1
        self.min_y = min(self.points[1])

        # find the number of points in each direction
        # needed for the array
        self.L_x = int(np.ceil((self.max_x-self.min_x)/ds))
        self.L_y = int(np.ceil((self.max_y-self.min_y)/ds))

        # get the coordinates for each direction
        xl = np.arange(self.min_x, self.max_x, ds)
        yl = np.arange(self.min_y, self.max_y, ds)
    
        # build the mask by checking if every point on a
        # meshgrid falls within the polygon
        xx, yy = np.meshgrid(xl, yl)
        coords = np.array([xx, yy])
        coords_linear = np.transpose(np.reshape(coords, (2, self.L_x*self.L_y)))
        mask_linear = np.array(np.where(self.poly.contains_points(coords_linear, radius=ds), 1, 0))
        mask = np.reshape(mask_linear, (self.L_x, self.L_y))
        
        # set the values at the edges of the mask to emulate
        # attenuation at the boundaries
        # checks if the points are along the edge by taking
        # the cross product, ie checking that
        #        AP x AB = 0
        #    => (P-A) x (B - A) = 0
        #    => ((Px - Ax)(By - Ay) - (Py - Ay)(Bx - Ax))k^ = 0
        #    => (Px - Ax)(By - Ay) = (Py - Ay)(Bx - Ax)
        #    => Px(By - Ay) + Py(Bx - Ax) = AxBy - AyBx
        #    => Pi . (B - A) = AxBy - AyBx
        #
        # TODO: Make this vectorized, if possible
        atten_mask_linear = []
        if attenuation:
            for i in range(len(coords_linear)):
                unset = True
                P = np.array(coords_linear[i])
                Pi = np.array([coords_linear[i][1], coords_linear[i][0]])
                for vert in range(len(vertices)-1):
                    A = np.array([vertices[vert][0], vertices[vert][1]])
                    B = np.array([vertices[vert + 1][0], vertices[vert+1][1]])
                    if abs(np.dot(Pi, B-A)) == abs(A[0]*B[1] - A[1]*B[0]) and np.sum((P-A)*(P-B)) <= 0:
                        atten_mask_linear.append(attenuation[vert])
                        unset = False
                        break
                if unset:
                    atten_mask_linear.append(0)
                    unset = True
        
        atten_mask = np.reshape(atten_mask_linear, (self.L_x, self.L_y))
        
        return mask + atten_mask
    
    
    def Plot_Mask(self):
        '''Plot the shape of the mask.'''
        import matplotlib.patches as pat
        fig, ax = plt.subplots()
        patch = PathPatch(self.poly, facecolor='green', lw=0)
        #ax.add_patch(patch)
        
        plt.imshow(self.mask)
        
    def __init__(self, vertices, ds: float, dt: float, tmax: float, G: float, C:float, *, attenuation_vals=None):
        self.max_t = tmax
        self.dt = dt
        self.L_t = int(np.ceil(self.max_t / self.dt))
        
        self.mask = self.Poly_Mask(vertices, ds, attenuation_vals)
        self.G = G
        self.C = C
        
        self.system = np.zeros((self.L_x, self.L_y, self.L_t), dtype=float)
    #def __init__(self, min_x, max_x, min_y, max_y)

def solve2D_t_vec(sys: Room_2D, drive, fx: int, fy: int):
    '''Solve the given system using a finite differences solver, and 
    return the solved system.'''
    t1 = timer()
    
    mask = sys.mask
    u = sys.system
    G = sys.G
    C = sys.C
    
    # apply initial mask 
    u[:,:,0] = np.multiply(u[:,:,0], mask)
    
    print('Solving...')
    for t in range(1,len(times)-1):
        u[:,:,t] = np.multiply(u[:,:,t], mask)
        #u[fx,fy,t] = f(times[t]) #* np.exp(-times[t]) 
        D2x = u[:-2,1:-1,t] - 2 * u[1:-1,1:-1,t] + u[2:,1:-1,t]
        D2y = u[1:-1,:-2,t] - 2 * u[1:-1,1:-1,t] + u[1:-1,2:,t]
        
        u[1:-1,1:-1,t+1] = C*(D2x + D2y) + 2*u[1:-1,1:-1,t] + (G-1)*u[1:-1,1:-1,t-1]
        u[fx,fy,t+1] += dt**2 * drive(times[t])
        
    u /= 1+G
    
    t2 = timer()
    print('Done. ({}s)'.format(round(t2-t1,3)))
    
    return u

def waves2d(sys, freq, fx, fy):
    
    # check if it satisfies CFL condition
    if 2*C > 1:
        print('error: 2C > 1')
        print('C =', C)
        return None
    
    # define the driving function to be used
    drive = lambda t : np.sin(freq * t * 2 * np.pi) 

    # calculate the solution to the system 
    soln = solve2D_t_vec(sys, drive, fx, fy)
    
    return soln

def run_simulation(vertices, frequency_range, attenuation = [1,0.8,0.6,0.4,0.2,0]):
    """
    vertices (list of tuples): vertices of the room (a closed polygon)
    attentuation (list of floats): variable frequency in center of room. values between 0 to 1
    frequency_range (list of ints): in asecending order
    """

    # define frequency range
    fi = frequency_range[0]
    ff = frequency_range[1]
    df = 0.1
    freqs = np.arange(fi, ff, df)

    # a) square

    # find the room and mask
    sys = Room_2D(vertices, ds, dt, tf, G, C, attenuation_vals=attenuation)

    # find the solution at each frequency step
    patterns = []
    for fr in freqs:
        # calculate the solution
        soln = waves2d(sys, fr, 10, 10)
        
        # assume the solution is stable past 2s, so take
        # latter half of solution
        stable = len(soln[0,0,:])//2
        
        # fidn the absolute value of the solution
        abs_mag = np.abs(soln[:,:,stable:])
        # multiply by 100 to reduce difference in peaks between
        # driven peak and resonant peaks
        abs_mag = np.multiply(abs_mag,100)
        # average the result over the second half of the array
        # to find the relative amplitudes and nodes of the solution
        avg = np.mean(abs_mag, 2)
        patterns.append(avg)

    return sys,patterns