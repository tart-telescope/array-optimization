# This file will contain the script for the mask that will be used in the optimasion of the antenna array code

import torch 

def mask(x):

    cy = x//2
    cx = x//2 
    center_position=[cx,cy]

    print(x)
        
    mask = torch.zeros(size=(x, x), dtype=torch.complex64, requires_grad=True )
    y, x = torch.meshgrid(torch.arange(x), torch.arange(x))
    mask = (x - cx)**2 + (y - cy)**2 <= 1**2

    return mask