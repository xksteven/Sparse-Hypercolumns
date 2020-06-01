# Sparse-Hypercolumns

This is an implementation of zoomout or hypercolumns in Pytorch as seen in papers such as Learning Representations for Automatic Colorization.

This is based off of the models in Torchvision 1.5.0.  

Currently implemented: 
Resnet

TODOs:
other models (densenet,...)

Passing in partial indices for hypercolumns

The Examples


    import models.resnet
    
    model = resnet18()
    model(torch.randn(1,3,256,256))
    logits, hypercolumns = model(x)

The hypercolumns can then be passed into another model or used.

Possible arguments:

    model = resnet18(full_hypercolumns=True)
    # or at runtime one can provide a partial or full list of indices one wants to use for the output
    model(x, indices=[[0,1,37],[2,5,49]], out_size=(50,50))
    # format of indices is [[row_indices], [column_indices]]

