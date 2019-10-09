# Sparse-Hypercolumns

This is an implementation of zoomout or hypercolumns in Pytorch as seen in papers such as Learning Representations for Automatic Colorization.

This is based off of the models in Torchvision 0.4.0.  

Currently implemented: 
Resnet

TODOs:
Densenet

Passing in partial indices for hypercolumns

The Examples


    import models.resnet
    
    model = resnet18()
    ...
    linear_output, hypercolumns = model(x)

The hypercolumns can then be passed into another model or used.

Possible arguments:

    model = resnet18(sparse=True)
    # or at runtime one can provide the indices one wants to use for the output
    model(x, indices=[[0,1],[2,5]], output_shape=(50,50))

Notes:

Currently length of indices should match the size of the output shape.

