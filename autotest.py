import torch
import numpy as np

class autotest():
    def __init__(self, model, mode, device):
        self.model = model
        self.mode = mode
        self.device = device
    
    
