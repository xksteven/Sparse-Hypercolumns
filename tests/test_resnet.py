import models.resnet as resnet
import numpy as np
import torch
import unittest


class test_resnet(unittest.TestCase):

    def test_creation(self):
        model = resnet.resnet50(out_size=32)

    def test_hypercolumn_square_outsize(self):
        model = resnet.resnet50(out_size=[32,32])
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

    def test_shape(self):
        indices = np.array([[0,1,10], [10,20,15]])
        x = torch.rand(1,3,224,224)
        model = resnet.resnet18(indices=[indices[0, :], indices[1, :]])
        res = model(x)
        assert res[1].size() == torch.Size([1, 963, 3])


if __name__ == "__main__":
    unittest.main()
