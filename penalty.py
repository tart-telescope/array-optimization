import torch


def penalize(duv, limit=0.25):
    sharpness = 1
    m =  torch.nn.Softplus()
    clip_lower = m((limit - duv)*sharpness)/sharpness
    return clip_lower/limit


def penalize_below(x, limit, width):
    # Scale so that duv is expressed in multiples of width
    xs = x / width
    ls = limit / width
    return 10*limit*penalize(xs, ls)/width


def penalize_above(x, limit=0.2):
    sharpness = 40
    m =  torch.nn.Softplus()
    clip_lower = m((x-limit)*sharpness)/sharpness   # Always positive
    ret = 100 * clip_lower
    return ret


def get_penalty(x,y,z, min_spacing, max_radius):
    num_ant = x.shape[0]

    penalty = 0
    num_ant = x.shape[0]
    
    ## Penalize antennas too close to each other
    for i in range(num_ant):
        for j in range(num_ant):
            if (i != j):
                u = x[i] - x[j]
                v = y[i] - y[j]
                w = z[i] - z[j]
                
                duv = torch.sqrt (u**2 + v**2 + w**2)
                penalty += penalize_below(duv, min_spacing, width=0.01)
    
    # Penalize antennas to far from the centre 
    for i in range(num_ant):
        penalty += penalize_above(x[i]*x[i] + y[i]*y[i], max_radius*max_radius)
        
    return penalty




def get_penalty_rect(x,y,z, min_spacing, ply_wood_xdim, ply_wood_ydim):
    num_ant = x.shape[0]

    penalty = 0
    num_ant = x.shape[0]


    ## Penalize antennas too close to each other
    for i in range(num_ant):
        for j in range(num_ant):
            if (i != j):
                u = x[i] - x[j]
                v = y[i] - y[j]
                w = z[i] - z[j]
                
                duv = torch.sqrt (u**2 + v**2 + w**2)
                penalty += penalize_below(duv, min_spacing, width=0.01)
    
    # Penalize antennas to far from the centre 
    for i in range(num_ant):
        
        penalty += penalize_above(torch.abs(x[i]), ply_wood_xdim/2)
        penalty += penalize_above(torch.abs(y[i]), ply_wood_ydim/2)
        
    return penalty







if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    
    p = torch.linspace(9, 11, 100)
    y = penalize_above(p, 10)
    plt.plot(p, y)
    plt.show()
