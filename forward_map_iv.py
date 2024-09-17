#
# Will require a package called 'disko'. This is installed using pip3 install disko
#
# Author Tim Molteno: tim@elec.ac.nz (c) 2024.
#
import logging
import torch
import numpy as np
import penalty
import json

GPS_FREQ = 1.57542e9

logger = logging.getLogger(__name__)

# from disko import SquareFoV
# '''
def elaz2lmn(el_r, az_r):
    l = torch.sin(az_r) * torch.cos(el_r)
    m = torch.cos(az_r) * torch.cos(el_r)
    # Often written in this weird way... np.sqrt(1.0 - l**2 - m**2)
    n = torch.sin(el_r)
    return l, m, n

def pixels_to_elevation(_dr, _width_pix, _width_rad):
    _theta_min = np.pi/2 - (_width_rad/2)
    _r = (_width_pix/2)/np.cos(_theta_min)
    
    theta = 0
    if (_dr <= _r):
        theta = np.arccos(_dr/_r)
    return theta

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
        
        
        # Phase center to edge is delta_el = width_rad/2
        # width_pix/2 = cos(d_el)
        # 
        width_arcmin = np.degrees(width_rad)*60
        # Actual width and resolution
        self.width_pix = 2*int(np.ceil(width_arcmin / res_arcmin))  # To account for nyquist we multiply by 2: pix / 2 = width / res   
        self.res_arcmin = 2*width_arcmin / self.width_pix
        
        
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
       
        # Calculate elevation and azimuth
        for i in range(self.width_pix):
            dx = (i - center)
            for j in range(self.width_pix):
                dy = j - center
                
                dr = np.sqrt(dx*dx + dy*dy)
                el = pixels_to_elevation(dr, _width_pix = self.width_pix, _width_rad=self.width_rad )
                
                az = np.arctan2(dy,dx)

                el_r.append(el)
                az_r.append(az)
        
        self.el_r = torch.tensor(el_r)
        self.az_r = torch.tensor(az_r)
        
        self.l, self.m, self.n = elaz2lmn(self.el_r, self.az_r)
        self.n_minus_1 = self.n - 1


    def get_mask(self, center_deg):
        ret = torch.ones(size=(self.npix,), dtype=torch.float)
        
        elevation_limit = np.pi/2 - np.radians(center_deg/2)
        
        for i in range(self.npix):
            el = self.el_r[i]
            
            if (el >= elevation_limit):
                ret[i] = 0.0
        
        return ret.reshape(self.width_pix, self.width_pix)
    
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
        exponent = (u*l + v*m + w*n_minus_1).type(torch.cfloat)*jomega
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
    
    sky = torch.zeros(size=(l.shape[0],), requires_grad = True)
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




def param2xyz(p):
    global n_ant
    return p.reshape(3, n_ant)

def loss_function(p):
    _x, _y, _z = param2xyz(p) # torch.tensor_split(p, 3)

    global fov, n_ant, radius, mask, inverse_mask, max_x, max_y
    min_spacing = 0.25
    
    penalty_score = penalty.get_penalty_rect(_x,_y,_z, min_spacing, max_x, max_y)
    image = point_spread_function(_x,_y,_z,fov)
    
    image_score = torch.max(image*mask)/torch.max(image)
    # print(image_score.detach().numpy(), penalty_score.detach().numpy())
    return  2*torch.log(penalty_score+1) + 10*image_score



if __name__=="__fmain__":
    import matplotlib.pyplot as plt

    width_pix = 280 
    width_rad = np.radians(140)
    
    dr = np.linspace(0, width_pix /2, 100)

    elev = np.array([pixels_to_elevation(x, width_pix, width_rad) for x in dr])
    
    plt.plot(dr, np.degrees(elev))
    plt.show()


if __name__=="__main__":
    import argparse
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser(description='Use torch to image the point spread function', 
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate.")
    parser.add_argument('--arcmin', type=float, default=60, help="Resolution of the sky in arc minutes.")
    parser.add_argument('--radius', type=float, default=1.5, help="Length of each arm in meters.")
    parser.add_argument('--radius-min', type=float, default=0.1, help="Minimum antenna position along each arm in meters.")
    parser.add_argument('--spacing', type=float, default=0.25, help="Minimum antenna spacing.")

    parser.add_argument('--fov', type=float, default=160.0, help="Field of view in degrees")

    np.seterr(all='raise')

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
    min_spacing = ARGS.spacing

    ### Create the fov object once. It should NOT be created each time the array is evaluated
    fov = SquareFoV(res_arcmin=ARGS.arcmin,
                    theta = 0.0, phi=0.0,
                    width_rad=np.radians(ARGS.fov))
        
    ## Generate some random antenna positions
    max_x = 1.05/2
    max_y = 2.25/2
    
    least_max_baseline = min(max_y*2, max_x*2)
    
    xs = torch.linspace(-max_x, max_x, 4)*0.9
    ys = torch.linspace(-max_y, max_y, 6)*0.9

    x, y = torch.meshgrid(xs, ys, indexing='xy')
    
    x = x.flatten() + torch.normal(0, max_x/20, size=(n_ant, ))
    y = y.flatten() + torch.normal(0, max_y/20, size=(n_ant, ))
    z = torch.zeros_like(x)

    # Create a mask
    resolution = np.degrees(1.22 * 0.21 / least_max_baseline) # Rayleigh criterion
    print(f"Resolution: {resolution}")
    mask = fov.get_mask(center_deg=resolution)
    inverse_mask = (-mask + 1)

    plt.imshow(mask)
    plt.title("Mask")
    plt.ylabel("Image Height")
    plt.xlabel("Image Width")
    plt.grid(color='gray', linestyle='-', linewidth=0.5)
    plt.savefig('mask.png')

    # Create a parameter vector from the positions
    x_opt = torch.hstack([x,y,z])


    plt.ion()
    figure, (ax1, ax2) = plt.subplots(figsize=(16, 8), ncols=2, layout = 'constrained')
    line1, = ax1.plot(x, y, 'o',color="black")
    ax1.set_aspect(1)
    ax1.grid(True)
    
    psf = point_spread_function(x.detach(),y.detach(),z.detach(), fov)
    im = ax2.imshow(psf.detach())
    plt.title("Initial Point Spread Function")
    plt.ylabel("Image Height")
    plt.xlabel("Image Width")
    figure.colorbar(im, ax=ax2, use_gridspec=True)

    line1, = ax1.plot(x, y, 'o',color="black")
    ax1.set_title("Initial Antenna Positions")
    ax1.set_ylabel("Array Height")
    ax1.set_xlabel("Array Width")
    ax1.grid(True)

    plt.savefig('beforecoordinates_RMSprop_125its_lr0.001_roof.png')

    #plt.show()
    plt.ioff()

    loss_history=[]

    opt = torch.optim.RMSprop([x_opt], lr=ARGS.lr)
    x_opt.requires_grad_()
    for i in range(125):
        _x, _y, _z = param2xyz(x_opt)
        line1.set_xdata(_x.detach())
        line1.set_ydata(_y.detach())


        psf = point_spread_function(_x.detach(),_y.detach(),_z.detach(), fov)
        im.set_data(psf.detach())
        # plt.savefig('after.png')
        
        loss = ((loss_function(x_opt)))

        if i % 10 == 0:
            print(f'i {i}: loss_history = {loss.item()}')

        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_history.append(loss.item())

        #print(f"loss: {loss.detach()}")
    
    
    plt.ion()
    figure, (ax1, ax2) = plt.subplots(figsize=(16, 8), ncols=2, layout = 'constrained')
    line1, = ax1.plot(_x.detach(), _y.detach(), 'o', color="black")
    ax1.set_aspect(1)
    ax1.grid(True)
    
    psf = point_spread_function(_x.detach(),_y.detach(),_z.detach(), fov)
    im = ax2.imshow(psf.detach())
    plt.title("Optimized Point Spread Function")
    plt.ylabel("Image Height")
    plt.xlabel("Image Width")
    figure.colorbar(im, ax=ax2, use_gridspec=True)

    line1, = ax1.plot(_x.detach(), _y.detach(), 'o',color="black")
    ax1.set_aspect(1)
    ax1.set_title("Optimized Antenna Positions")
    ax1.set_ylabel("Array Height")
    ax1.set_xlabel("Array Width")
    ax1.grid(True)

    plt.savefig('aftercoordinates_RMSprop_125its_lr0.001_roof.png')

    #plt.show()
    plt.ioff()
    
 

    #Track and grqph Loss Histeory
    plt.figure()
    plt.plot(loss_history, linestyle="-", color='b')
    plt.title("Loss History")
    plt.xlabel("Iterations")
    plt.ylabel("loss")
    plt.grid(color='gray', linestyle='-', linewidth=2.0)
    plt.savefig('LH-_RMSprop_125its_lr0.001_roof.png')
    #plt.show()


    # Save coordinates to JSON file

xlist = _x.detach().tolist()
ylist = _y.detach().tolist()
zlist = _z.detach().tolist()

ant_pos = [ [x,y,z] for x,y,z in zip(xlist, ylist, zlist)]
coordinates_1 = {
    "x_opt": xlist,
    "y_opt": ylist,
    "z_opt": zlist,
    "ant_pos" : ant_pos
}

filename = 'coordinates_RMSprop_125its_lr0.001_roof.json'
with open(filename, 'w') as json_file:
    json.dump(coordinates_1, json_file, indent=5)

print(f"Coordinates have been saved to '{filename}'")


    


        
