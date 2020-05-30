import logging
import numpy as np
import torch
import torch.nn as nn

from typing import Tuple, List, Optional

TensorList = List[torch.Tensor]
# IndexTuple = Tuple(np.array, np.array)

def calculate_hyper_indices(in_planes: torch.Tensor, out_planes:torch.Tensor, prev_indices: np.array) -> np.array:
    _, _, in_h, in_w = in_planes.size()
    _, _, out_h, out_w = out_planes.size()
    rows, cols = prev_indices
    # center the rows (or cols), then scale, then back to 0 to height (or width)
    new_rows = ((rows - in_h//2) * (out_h/in_h) + out_h//2).astype(np.int)
    new_cols = ((cols - in_w//2) * (out_w/in_w) + out_w//2).astype(np.int)
    return np.asarray(new_rows, new_cols)


def get_random_indices(in_size: Tuple[int,int], out_size: Tuple[int,int], indices: np.array) -> np.array:
    total_size = (in_size[0]*in_size[1])
    elif len(indices) < total_size:
    elif len(indices) > total_size:
        #TODO raise warning for this case
        logging.warning(f"Number of indices provided {len(indices)}" 
                        "is greater than total out_size.")
        indices = indices[:total_size]    
    # sorted to keep some semblance of the original spatial information
    # might not be necessary
    indices = np.asarray(sorted(np.random.uniform(high=in_size[0], size=out_size[0] ), 
                     np.random.uniform(high=in_size[1], size=out_size[1])))
    return indices
    

class Hypercolumns(nn.Module):

    def __init__(self, in_size: Optional[Tuple[int,int]]=None, out_size: Tuple[int,int]=(32,32), 
                 full: boolean=False, indices: Optional[np.array]=None, 
                 interp_mode: str="bilinear", which_layers: Optional[Tuple[int, ...]]=None):
        """
        
        Arguments:
            in_size: 
            out_size: 
            full: 
            indices: 
            interp_mode: 
            which_layers: Allows for selection of subset of passed in layers.
        """
        super(Hypercolumns, self).__init__()
        
        if not indices and not out_size and not full:
            print("Please provide either out_size or full.")
            raise
        self.full = full
        
        if in_size is not None:

                indices = get_random_indices(in_size, out_size, indices)

        self.in_size = in_size
        self.out_size = out_size
        self.indices = indices
        self.interp_mode = interp_mode
        self.which_layers = which_layers
        # Variable that is computed after first call
        #TODO consider unsetting if incoming images will vary in dimensions
        self._index_list = None

    # be able to pass in a shape for hypercols
    # check if shape is bigger than any requested layer
    # scale layer to appropriate dimension and take full layer
    # Need option for full hypercolumn



    def cat_features(self, x: TensorList):
        return torch.cat(x, dim=1)

    
    # The main method should take in a list of the features and return the hypercolumns
    def create_hypercolumns(self, hyperlist: TensorList) -> torch.Tensor:
        """
        List of Torch tensors
        """

        if self.full:
            for index, layer in enumerate(hyperlist):
                hyperlist[index] = nn.functional.interpolate(layer, self.out_size, mode= self.interp_mode)
            hypercols = torch.cat(hyperlist, dim=1)
            return hyercols
        else:
            if self.in_size is None:
                # Assumes BCHW format
                self.in_size = hyperlist[0].size()[-2:]
                #TODO not done
                indices_list = [self.]

            # calculate the indices for each layer
            for index, layer in enumerate(hyperlist):
                if (index == (len(hyperlist) - 1)):
                    break
                prev_indices = indices_list[-1]
                in_planes = hyperlist[index]
                out_planes = hyperlist[index+1]
                indices_list.append(calculate_hyper_indices(in_planes, out_planes, prev_indices))

            for index, layer in enumerate(hyperlist):
                rows, cols = indices_list[index]
                hyperlist[index] = hyperlist[index][:,:,rows,cols]

            hypercols = torch.cat(hyperlist, dim=1)
            return hypercols
