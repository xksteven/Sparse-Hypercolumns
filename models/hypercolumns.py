import logging
import numpy as np
import torch
import torch.nn as nn

from typing import Tuple, List

TensorList = List[torch.Tensor]

def calculate_hyper_indices(in_planes: torch.Tensor, out_planes:torch.Tensor, prev_indices: np.array):
    _, _, in_h, in_w = in_planes.size()
    _, _, out_h, out_w = out_planes.size()
    rows, cols = prev_indices
    # center the rows (or cols), then scale, then back to 0 to height (or width)
    new_rows = ((rows - in_h//2) * (out_h/in_h) + out_h//2).astype(np.int)
    new_cols = ((cols - in_w//2) * (out_w/in_w) + out_w//2).astype(np.int)
    return (new_rows, new_cols)

def get_random_indices(out_size: Tuple, indices: np.array):
    out_size = (output_shape[0]*output_shape[1], 2)
    indices = sorted(np.random.uniform(high=x.size()[2], size=out_size))

class Hypercolumns(nn.Module):

    def __init__(self, out_size: Tuple=(32,32), full: boolean=False, indices: np.array=None):
        """
        
        Arguments:
            out_size:
            full: 
            indices:
        """
        super(Hypercolumns, self).__init__()
        
        if not indices and not out_size and not full:
            print("Please provide either out_size or full.")
            raise
        self.full = full
        
        total_size = (out_size[0]*out_size[1])
        elif len(indices) < total_size:
            indices = get_random_indices(out_size, indices)
        elif len(indices) > total_size:
            #TODO raise warning for this case
            logging.warning(f"Number of indices provided {len(indices)}" 
                            "is greater than total out_size.")
            indices = indices[:total_size]
        self.out_size = out_size
        self.indices = indices

    # be able to pass in a shape for hypercols
    # check if shape is bigger than any requested layer
    # scale layer to appropriate dimension and take full layer
    # Need option for full hypercolumn


    # this should work layer by layer
    # take in indices and return hyper column

    def get_hypercolumn(self):
        if self.full:
            pass
        elif self.sampling:
           if indices == None:
           indices_list = [indices]

           hypercols = [x1, x6, x7, x8]

           # calculate the indices for each layer
           for index, l in enumerate(hypercols):
               if (index == (len(hypercols) - 1)):
                   break
               prev_indices = indices_list[-1]
               in_planes = hyper_cols[index]
               out_planes = hyper_cols[index+1]
               indices_list.append(calculate_hyper_indices(in_planes, out_planes, prev_indices))

           for index, l in enumerate(hypercols):
               rows, cols = indices_list[index]
               hypercols[index] = hypercols[index][:,:,rows,cols]

           hypercols = torch.cat(hypercols, dim=1)

        # else we take the full feature maps as our hypercolumns
        else:
           hypercols = [x1, x6, x7, x8]
           for index, l in enumerate(hypercols):
               hypercols[index] = nn.functional.interpolate(l, output_shape, mode= interp_mode)


    def cat_features(self, x: TensorList):
        return torch.cat(x, dim=1)

    # The main method should take in a list of the features and return the hypercolumns
    def create_hypercolumns(self, x: TensorList):
        """
        List of Torch tensors
        """


