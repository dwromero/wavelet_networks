import torch
import numpy as np

## Returns the support of the 1D cardinal B-spline in terms of a min-max range
def B_supp(n, s=1, dx=0, intsupp=False):
    """ Returns a min and max value of the domain on which the 1D cardinal B-spline of order n is non-zero.

        INPUT:
            - degree n, an integer

        INPUT (optional):
            - scale s, a real scalar number. Specifies the support of scaled B-splines via supp( B( . / s) )
            - offset dx, a real scalar number. Specifies the support of scaled+shifted B-splines via supp(B( . / s - dx)
            - intsupp, a boolean. Specifies whether or not the support should be on an integer grid. E.g. if xMax would
              be 2.3, and we only sample integer positions x. Then 2 would still be non-zero, but 3 would evaluate to
              zero. In this case the non-zero interval would be [-2,2] whereas in the intsupp=False case it would be
              [-2.3,2.3]

        OUTPUT:
            - (xMin, xMax), the min-max range of the support
    """
    xMinMax = s * (n + 1) / 2
    xMin = -xMinMax + dx
    xMax = xMinMax + dx
    if intsupp:
        xMax = (int(xMax) - 1 if int(xMax) == xMax else int(xMax))
        xMin = (int(xMin) + 1 if int(xMin) == xMin else int(xMin))
    return (xMin, xMax)


## Returns the grid (1D torch tensor) with unit gridpoint spacing
def B_supp_grid(n, s=1, dx=0, intsupp=False, device='CPU'):
    """ Returns a grid (1D torch tensor) with unit spacing between the grid points (e.g. [xMin,...,-1,0,1,...,xMax]).
        The min-max range is computed via B_supp.

        INPUT:
            - degree n, an integer

        INPUT (optional):
            - scale s, a real scalar number. Specifies the support of scaled B-splines via supp( B( . / s) )
            - offset dx, a real scalar number. Specifies the support of scaled+shifted B-splines via supp(B( . / s - dx)
            - intsupp, a boolean. Specifies whether or not the support should be on an integer grid. E.g. if xMax would
              be 2.3, and we only sample integer positions x. Then 2 would still be non-zero, but 3 would evaluate to
              zero. In this case the non-zero interval would be [-2,2] whereas in the intsupp=False case it would be
              [-2.3,2.3]

        OUTPUT:
            - xx, a 1D torch.tensor of x-values for which B(x) is non-zero
    """
    xMin, xMax = B_supp(n, s, dx, intsupp)      # TODO: With intsupp=False, I get [-1, 0, 1, 2]. But i think it should be symmetrical. Right?
    return torch.arange(xMin,xMax+1,dtype=torch.int16,device=device)     #TODO device not requried. Managed automatically by model.device().


## Returns the grid (1D torch tensor) with unit gridpoint spacing
def B_supp_grid_2(n, s=1, intsupp=False, device='CPU'):
    """ Returns a grid (1D torch tensor) with unit spacing between the grid points (e.g. [xMin,...,-1,0,1,...,xMax]).
        The min-max range is computed via B_supp.

        INPUT:
            - degree n, an integer

        INPUT (optional):
            - scale s, a real scalar number. Specifies the support of scaled B-splines via supp( B( . / s) )
            - offset dx, a real scalar number. Specifies the support of scaled+shifted B-splines via supp(B( . / s - dx)
            - intsupp, a boolean. Specifies whether or not the support should be on an integer grid. E.g. if xMax would
              be 2.3, and we only sample integer positions x. Then 2 would still be non-zero, but 3 would evaluate to
              zero. In this case the non-zero interval would be [-2,2] whereas in the intsupp=False case it would be
              [-2.3,2.3]

        OUTPUT:
            - xx, a 1D torch.tensor of x-values for which B(x) is non-zero
    """
    xMin, xMax = B_supp(n, s, 0, intsupp)      # TODO: With intsupp=False, I get [-1, 0, 1, 2]. But i think it should be symmetrical. Right?
    return xMin, xMax, torch.arange(xMin,xMax+1,dtype=torch.int16, device=device)


if __name__ == '__main__':
    from eerie.bsplines.b_1d import B
    n = 3
    s = 1
    dx=0.2
    Bfunc = B(3)
    xlist = B_supp_grid(n, s, dx, True)
    print(B_supp(n, s, dx))
    print(xlist)
    print(Bfunc((xlist - dx) / s))
