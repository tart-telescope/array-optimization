import torch
import matplotlib.pyplot as plt
import numpy as np


def loss_function (area, perimeter_penalty):
    
    return -area + perimeter_penalty**1
