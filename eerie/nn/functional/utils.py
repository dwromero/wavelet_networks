import torch
from eerie.bsplines.b_1d import B
from eerie.bsplines.utils import B_supp_grid


## Some util functions
# Tested agains implementation based on torch.roll which is slightly slower, but not too much actually
def cropped_translate(input, shift, crop, dim=None, stride=1):
    # Creates a view instead of copy (as done with torch.roll)
    # Rolls over the spatial dimensions by slicing the input tensor
    inshape = input.shape
    if dim is None:
        d = len(inshape) - 2  # minus batch axis, minus channel axis
        slices = [...] + [slice(crop[0] + shift[i], inshape[2 + i] - crop[1] + shift[i], stride) for i in range(d)]
    else:
        slices = [slice(None, None, None)] * len(inshape)
        slices[2 + dim] = slice(crop[0] + shift, inshape[2 + dim] - crop[1] + shift, stride)
    return input[slices]


def bcropped_translate(input, n, s, x, crop, stride=1):
    # Translates by scalar valued shifts, this requires interpolation which we do via B-splines. Essentially we perform
    # shifts by convolutions with shifted cardinal B-splines of order n and scale s. In order for the shift via cropped_
    # translate to work it is required that "crop" >= than the maximum shift + the B-spline support.
    # See e.g. eerie.nn.functional.bconv.
    d = len(input.shape) - 2  # Number of spatial dimensions
    # The cardinal b-spline
    Bfunc = B(n)

    # The B-spline kernels as 1D sparse conv kernels
    # TODO: Probably unnecessary load on the GPU (or use fixed size splines and do an einsum, or something)
    # xint = x.type(torch.int16)
    # dx = x - xint
    # integer_grid_pts = [B_supp_grid(n, s, dx[dim], True, x.device) + xint[dim] for dim in range(d)]
    # integer_grid_vals = [Bfunc((integer_grid_pts[dim] - x[dim]) / s) for dim in range(d)]
    brange = B_supp_grid(n, s, 0, True, x.device)
    integer_grid_pts = (x[:,None] + torch.arange(brange[0],brange[-1]+1).to(x.device)[None,:]).round()      # TODO Unsqueeze instead of [:,None]
    integer_grid_vals = Bfunc( ((integer_grid_pts - x[:,None]) / s) )

    # Separable convolution (or B-spline interpolation)
    output = input
    for dim in range(d):
        output = sum([integer_grid_vals[dim][j] * cropped_translate(output, int(integer_grid_pts[dim][j]), crop, dim=dim,
                                                                    stride=stride) for j in
                      range(integer_grid_vals[dim].shape[0])])
    # Return output
    return output
