import models.resnet as resnet
import torch
import unittest


class test_resnet(unittest.TestCase):

    def test_creation(self):
        model = resnet.resnet50()

    def test_hypercolumn(self):
        model = resnet.resnet50()
        res = model(torch.randn(1, 3, 256, 256))

if __name__ == "__main__":
    unittest.main()
