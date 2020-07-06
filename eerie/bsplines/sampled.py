import torch as torch
import math as m
from eerie.bsplines.b_dd import B


# Sample a B-spline that consists of a sum of shifted cardinal B-splines
def b_spline(n, s, weights, centers, size, centered = True):
    d = centers.shape[1]
    x_grid = grid(size, d, centered=centered, flatten=False)
    B_func = B(n)
    # weights_reshaped = 0
    sampled = sum([weights[i][[...]+[None]*d]*B_func((x_grid - centers[i])/s)[None,...] for i in range(centers.shape[0])])
    return sampled


## Convolution kernels (via tensor product of 1D kernels)
def cardinal_b_spline(n, s, size, dim):
    x_grid_1d = grid(size, 1, centered=True)
    kernel_1d = B(n)(x_grid_1d / s)
    # Repeated tensor product
    kernel_dd = tensor_power(kernel_1d, dim)
    # Return the d-dimensional kernel
    return kernel_dd


# (Dual) Kernel
def cardinal_b_spline_dual(n, s, size, dim):
    dual_kernel_1d = cardinal_b_spline_dual_1d(n, s, size)
    dual_kernel_dd = tensor_power(dual_kernel_1d, dim)
    return dual_kernel_dd


def cardinal_b_spline_dual_1d(n, s, size):
    # Construct the 1D sample grid (xGrid) and the grid of 1D B-spline centers (xiGrid)
    x_grid_1d = grid(size, 1, centered=True)
    Ni = m.ceil(size / s)  # Nr of basis functions per axis
    if Ni % 2 == 0:
        Ni += 1
    xi_grid_1d = grid(Ni, 1, centered=True, flatten=True) * s
    # For each xi sample the basis functions on the grid
    basis_1d = torch.stack([B(n)((x_grid_1d - xi) / s) for xi in xi_grid_1d], 0)  # .shape = (Ni,size,size)
    # Construct the Gramm matrix, invert it, and then construct the dual basis
    gg_1d = torch.tensordot(basis_1d, basis_1d, (list(range(-1, 0)), list(range(-1, 0))))  # .shape = (Ni,Ni)
    basis_dual_1d = torch.tensordot(torch.inverse(gg_1d), basis_1d, ([-1], [0]))
    # The central dual basis vector is our estimated cardinal dual basis
    dual_kernel_1d = basis_dual_1d[Ni // 2]
    return dual_kernel_1d


def tensor_power(tensor_1d, d):
    tensor_dd = tensor_1d
    for i in range(1, d):
        tensor_dd = tensor_dd[..., None] * tensor_1d[None, ...]
    return tensor_dd


## Grid functions used in the kernel functions
# d-Dimensional grid
def grid(size, dim, centered=False, flatten=False):
    # For some reason the torch.meshgrid ordering is different to the ordering of np.meshgrid
    grid = torch.stack(torch.meshgrid(*[grid_1d(size, centered=centered)] * dim)[::-1], -1)
    if flatten:
        return torch.reshape(grid, (-1, dim))
    else:
        return grid


# 1-Dimensional grid
def grid_1d(size, centered=False, flatten=False):
    if centered:
        grid = torch.linspace(-m.floor(size / 2), m.floor(size / 2), 2 * m.floor(size / 2) + 1)
    else:
        grid = torch.linspace(0, size - 1, size)
    return grid

if __name__ == '__main__':

    from eerie.bsplines.b_dd import B
    from eerie.bsplines.utils import B_supp
    xMax = 5
    d = 2
    N = 3
    Cin = 3
    Cout = 5

    n = 1
    s = 2

    _ , xMaxB = B_supp(n, s)
    size = 2*(xMax + xMaxB)

    centers = (torch.rand(2 * N, d) - 0.5) * 2 * xMax
    centers = torch.randint(-xMax, xMax, [N, d])
    weights = torch.randn(N, Cin, dtype=torch.float32)

    d = centers.shape[1]
    x_grid = grid(size, d, centered=True, flatten=False)
    B_func = B(n)
    sampled = b_spline(n, s, weights, centers, size, centered=True)

    from matplotlib import pyplot as plt
    plt.imshow(sampled[0,:,:])
    plt.show()
