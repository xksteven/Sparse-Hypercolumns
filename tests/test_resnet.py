import models.resnet as resnet
import numpy as np
import torch
import unittest


class test_resnet(unittest.TestCase):

    def test_creation(self):
        model = resnet.resnet50()

    def test_hypercolumn(self):
        model = resnet.resnet50()
        res = model(torch.randn(1, 3, 256, 256))
    
    def test_fullhypercolumn(self):
        model = resnet.resnet50(full_hypercolumns=True)
        res = model(torch.randn(1, 3, 256, 256))

    def test_indices(self):
        model = resnet.resnet50(indices=np.array(
            [[0, 1, 37], 
             [100, 14, 101]]))
        res = model(torch.randn(1, 3, 256, 256))

    def test_indices_list(self):
        model = resnet.resnet50(indices=[[0, 1, 37], 
             [100, 14, 101]])
        res = model(torch.randn(1, 3, 256, 256))



if __name__ == "__main__":
    unittest.main()
