#
# Will require a package called 'disko'. This is installed using pip3 install disko
#
# Author Tim Molteno: tim@elec.ac.nz (c) 2024.
#
import logging
import torch

import numpy as np

import penalty

GPS_FREQ = 1.57542e9

logger = logging.getLogger(__name__)

# from disko import SquareFoV
# '''
def elaz2lmn(el_r, az_r):
    l = np.sin(az_r) * np.cos(el_r)
    m = np.cos(az_r) * np.cos(el_r)
    # Often written in this weird way... np.sqrt(1.0 - l**2 - m**2)
    n = np.sin(el_r)
    return l, m, n


class SquareFoV():

    def __init__(self, res_arcmin=None, theta=0.0, phi=0.0, width_rad=0.0):
        super().__init__()
        logger.info(r"SquareFoV:")
        logger.info(f"    res={res_arcmin} arcmin,")
        logger.info(f"    theta={theta}, phi={phi}, width={width_rad} rad)")
        # Theta is co-latitude measured southward from the north pole
        # Phi is [0..2pi]

        self.width_rad = width_rad

        self.theta = theta
        self.phi = phi
        
        # Actual width and resolution
        self.width_pix = 2*int(np.ceil(np.degrees(width_rad)*60 / res_arcmin))
        self.res_arcmin = np.degrees(width_rad)*60 / self.width_pix
        
        self.npix = self.width_pix*self.width_pix

        logger.info(
            f"New SubFoV, width_pix={self.width_pix} npix={self.npix}, res={self.res_arcmin} arcmin")

        self.fov = width_rad
        self.pixels = np.zeros(self.npix)

        self.pixel_areas = np.ones(self.npix)/self.npix

        # Assume flat sky
        # Create the direction cosines
        el_r = []
        az_r = []
        center = self.width_pix // 2
       

        arcmin2rad = np.radians(1.0/60)
        def pixels_to_rad(pix):
           return (pix*self.res_arcmin*arcmin2rad) / 2  # Nyquist
       
        # TODO add self.el_r and self.az_r
        for i in range(self.width_pix):
            dx = (i - center)
            for j in range(self.width_pix):
                dy = j - center
                
                dr = np.sqrt(dx*dx + dy*dy)
                el = np.pi/2 - pixels_to_rad(dr)
                
                az = np.arctan2(dy,dx)

                el_r.append(el)
                az_r.append(az)
        
        self.el_r = np.array(el_r)
        self.az_r = np.array(az_r)
        
        self.l, self.m, self.n = elaz2lmn(self.el_r, self.az_r)
        self.n_minus_1 = self.n - 1


    def __repr__(self):
        return f"SquareFoV fov={self.fov}, width={self.width_pix}, res={self.res_arcmin}"

    def to_hdf(self, filename):
        with h5py.File(filename, "w") as h5f:
            self.to_hdf_header(h5f)

            h5f.create_dataset('res_arcmin', data=[self.res_arcmin])
            h5f.create_dataset('theta', data=[self.theta])
            h5f.create_dataset('phi', data=[self.phi])
            h5f.create_dataset('width_rad', data=[self.width_rad])

            h5f.create_dataset('pixels', data=self.pixels)
            h5f.create_dataset('pixel_indices', data=self.pixel_indices)

    def get_image(self):
        return self.pixels.reshape(self.width_pix, self.width_pix)

# '''

def get_uvw(x,y,z):

    num_ant = x.shape[0]
    
    uvw = []
    indices = []
    for i in range(num_ant):
        for j in range(num_ant):
            if (i != j):
                u = x[i] - x[j]
                v = y[i] - y[j]
                w = z[i] - z[j]
                
                uvw.append([u,v,w])
                indices.append([i,j])
    return uvw, indices


def get_fringes(uvw, l,m,n_minus_1,freq=GPS_FREQ):
    C = 3e8
    wavelength = C / freq
    omega = 2.0*np.pi/wavelength
    jomega = 1.0j*omega
    pixel_areas = 4*np.pi/l.shape[0]
    fringes = []
    for u,v,w in uvw:
        exponent = (u.detach()*l + v.detach()*m + w.detach()*n_minus_1).type(torch.complex128)*jomega
        h = torch.exp(exponent) * pixel_areas
        fringes.append(h)
    return fringes


def get_gamma(uvw, l,m,n_minus_1, freq=1.57542e9):
    fringes = get_fringes(uvw, l,m,n_minus_1,freq=GPS_FREQ)
    gamma = torch.stack(fringes, axis=0)
    return gamma


def disko_image(vis, uvw, l,m,n_minus_1,freq=GPS_FREQ):
    # l, m, n are the co-ordinates in the sky. There is one for each pixel in the sky.
    # vis: The visibilities
    # u,v,w the uv coordinates
    
    sky = torch.zeros(size=(l.shape[0],))
    fringes = get_fringes(uvw, l,m,n_minus_1,freq=GPS_FREQ)
    for h,v in zip(fringes, vis):
        sky = sky + h*v
        
    return torch.abs(sky)



def point_spread_function(x,y,z, fov):
    uvw, indices = get_uvw(x,y,z)
    vis = np.ones(shape=(len(indices),))
    
    sky = disko_image(vis=vis, uvw=uvw, l=fov.l, m=fov.m, n_minus_1=fov.n_minus_1)
    image = sky.reshape(fov.width_pix, fov.width_pix)

    return image



def global_f(x):
    '''
        A function suitable for optimizing using a global minimizer. This will
        return the condition number of the telescope operator
    '''
    global fov, min_spacing, max_radius
    
    xy = x.reshape((2,-1))
    x = xy[0,:]
    y = xy[1,:]
    z = torch.zeros_like(x)

    image = point_spread_function(x,y,z,fov)
    
    ## Now score the image using torch functions.
    position_penalty = penalty.get_penalty(x,y,z, min_spacing, max_radius)

    # Insert a score for how good the point spread function is There
    image_penalty = 3
    
    return position_penalty + image_penalty


def global_f(p):
    _x, _y, _z = torch.tensor_split(p, 3)

    global fov
    min_spacing = 0.25
    radius = 1.7
    
    _penalty = penalty.get_penalty(_x,_y,_z, min_spacing, radius)
    image = point_spread_function(_x,_y,_z,fov)
    return _penalty

if __name__=="__main__":
    import argparse
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser(description='Use torch to image the point spread function', 
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    parser.add_argument('--arcmin', type=float, default=60, help="Resolution of the sky in arc minutes.")
    parser.add_argument('--radius', type=float, default=2.0, help="Length of each arm in meters.")
    parser.add_argument('--radius-min', type=float, default=0.1, help="Minimum antenna position along each arm in meters.")
    parser.add_argument('--spacing', type=float, default=0.15, help="Minimum antenna spacing.")

    parser.add_argument('--fov', type=float, default=140.0, help="Field of view in degrees")


    ARGS = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename='array_opt.log',
                        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    
    # add the handler to the root logger
    logger = logging.getLogger()
    logger.addHandler(console)


    n_ant = 24
    radius = ARGS.radius
    min_spacing = 0.2

    ### Create the fov object once. It should NOT be created each time the array is evaluated
    fov = SquareFoV(res_arcmin=ARGS.arcmin,
                    theta = 0.0, phi=0.0,
                    width_rad=np.radians(ARGS.fov))
    
    
    ## Generate some random antenna positions
    
    x = 2*radius*torch.rand(size=(n_ant,)) - radius
    y = 2*radius*torch.rand(size=(n_ant,)) - radius
    z = torch.zeros_like(x)
    
    # Score the image
    image = point_spread_function(x,y,z,fov)
    
    ## Now score the image using torch functions.
    
    position_penalty = penalty.get_penalty(x,y,z, min_spacing, radius)
    
    plt.imshow(image)
    # plt.show()
    
    x_opt = torch.hstack([x,y,z])
    
    
    opt = torch.optim.SGD([x_opt], lr=0.001, momentum=0)
    x_opt.requires_grad_()
    opt.zero_grad()

    for i in range(10):
        print(f"x_opt = {x_opt}")
        result = global_f(x_opt)
        loss = result
        loss.backward()
        opt.step()

