# torch
import torch
#built-in
import math
import numpy as np
# project
from eerie.nn.functional import gconv_Rd_G, gconv_G_G


class GConvRdG(torch.nn.Module):
    def __init__(self,
                 group,
                 in_channels,
                 out_channels,
                 kernel_size,
                 h_grid,
                 n_basis=None,
                 b_order=2,
                 b_scale=1.0,
                 adaptive_basis=False,
                 stride=1,
                 padding=0,
                 b_padding=True,
                 b_groups=1,
                 b_groups_sigmas=None,
                 bias=True,
                 virtual_kernel_size=None,
                 dilation=1):
        """ Generates a d-dimensional convolution layer with B-spline convolution kernels.
        Args:
            - d: int. The dimensionality of the input tensor (which will be of shape=[B,Cin,X_1,X_2,...,X_d] with X_i the size of each spatial axis).
            - in_channels: the number of input channels (Cin) of the input tensor.
            - out_channels: the number of output channels (Cout) of the to-be-generated output tensor.
            - kernel_size: int. This is the (virtual) kernel size of the convolution kernel.
        Args (optional):
            - n_basis: int or None. The number of non-zero weights in the convolution kernel. If not specified a dense convolution kernel will be used and n_basis=size**d. Otherwise the kernel of size kernel_size**d will have only n_basis number of non-zero weights. The indices of these locations will be randomly (uniform) initialized.
            - b_order: int. Order of the B-spline basis.
            - b_scale: float. Scale of the cardinal B-splines in the basis.
            - stride: int. not implemented...
            - padding: int. Integer that specifies the amount of spatial padding on each side.
            - b_padding: boolean. Whether or not to automatically correct for cropping due to the size of the cardinal B-splines.
            - b_groups: int. Split the convolution kernels (along the output channel axis) into groups that have their own set of centers (basis functions). If b_groups=out_channels, then each output channel is generated with a kernel that has its own basis consisting of n_basis functions.
            - bias: not implemented
        Output (of the generated layer):
            - output: torch.tensor, size=[B,Cout,X_1',X_2',...,X_d']. Here X_i' are the cropped/padded spatial dims.
        """
        # nn.Module init
        super(GConvRdG, self).__init__()

        # The base settings
        self.group = group
        self.d = group.Rd.d
        self.Cin = in_channels
        self.Cout = out_channels
        self.size = kernel_size
        self.h_grid = h_grid
        # self.xMax = kernel_size // 2  # Specifies the maximum allowed pixel offset
        self.N = n_basis
        # The convolution settings
        self.padding = padding
        self.b_padding = b_padding
        self.stride = stride
        self.b_groups = b_groups
        self.b_groups_sigmas = b_groups_sigmas
        # The B-spline settings
        self.n = b_order
        self.s = b_scale
        self.adaptive_basis = adaptive_basis
        self.virtual_kernel_size = virtual_kernel_size
        self.dilation = dilation

        if bias:
            # Random (normal dist) init
            self.bias = torch.nn.Parameter(torch.randn(out_channels).float())
        else:
            self.register_parameter('bias', None)

        # The indices (random if not specified, otherwise on a regular grid > dense kernel)
        centers, self.N, self.x_min, self.x_max = x_centers_init(self.d, self.size, self.N, b_groups=b_groups, integer=False)
        # If also optimizing over the centers:
        if self.adaptive_basis:
            self.centers = torch.nn.Parameter(centers.float())
        else:
            # self.register_buffer('centers',centers.float())
            self.centers = centers # .float()

        # Kaiming init
        # self.weights = torch.nn.Parameter(weights_init(self.N, self.Cout, self.Cin).float())
        # Check after checking
        self.weights = torch.nn.Parameter(torch.Tensor(self.Cout, self.Cin, self.N))    # [N_out, N_h, N_in, N_h, X, Y] is the standard form in torch.
        # Initialization parameters
        wscale = math.sqrt(2.)  # This makes the initialization equal to that of He et al
        self._reset_parameters(wscale=wscale)

        if self.virtual_kernel_size is None:
            max_scaling = torch.max(self.group.H.scaling(h_grid.grid))
            max_center = torch.max(torch.abs(self.centers))
            self.virtual_kernel_size = int(torch.round(max_scaling*max_center))*2 + 1

    def forward(self, input):
        output = gconv_Rd_G(input, self.weights, self.centers, self.virtual_kernel_size, self.group, self.h_grid, self.n, self.s, self.stride, self.padding,
                            self.b_padding, self.b_groups, self.bias, self.dilation)
        return output

    def _reset_parameters(self, wscale):
        n = self.Cin
        k = self.size ** self.d
        n *= k
        stdv = wscale * (1. / math.sqrt(n))
        self.stdv = stdv
        self.weights.data.uniform_(-stdv, stdv)

class GConvGG(torch.nn.Module):
    def __init__(self,
                 group,
                 in_channels,
                 out_channels,
                 kernel_size,
                 h_grid,
                 h_grid_in=None,
                 n_basis=None, # None: dense, integer: random global. Othwerise if n_basis_x or _h are specified then n_basis = (n_basis_x**Rd.d)*(n_basis_h**H.d)
                 n_basis_x=None, # None: dense on the kernel_size (so n_basis_x=kernel_size**Rd.d)
                 n_basis_h=None, # None:
                 b_order=2,
                 b_scale=1.0,
                 adaptive_basis=False,
                 stride=1,
                 padding=0,
                 b_padding=True,
                 b_groups=1,
                 b_groups_sigmas=None,
                 bias=True,
                 virtual_kernel_size = None,
                 dilation=1,
                 h_crop=False):
        """ Generates a d-dimensional convolution layer with B-spline convolution kernels.
        Args:
            - d: int. The dimensionality of the input tensor (which will be of shape=[B,Cin,X_1,X_2,...,X_d] with X_i the size of each spatial axis).
            - in_channels: the number of input channels (Cin) of the input tensor.
            - out_channels: the number of output channels (Cout) of the to-be-generated output tensor.
            - kernel_size: int. This is the (virtual) kernel size of the convolution kernel.
        Args (optional):
            - n_basis: int or None. The number of non-zero weights in the convolution kernel. If not specified a dense convolution kernel will be used and n_basis=size**d. Otherwise the kernel of size kernel_size**d will have only n_basis number of non-zero weights. The indices of these locations will be randomly (uniform) initialized.
            - b_order: int. Order of the B-spline basis.
            - b_scale: float. Scale of the cardinal B-splines in the basis.
            - stride: int. not implemented...
            - padding: int. Integer that specifies the amount of spatial padding on each side.
            - b_padding: boolean. Whether or not to automatically correct for cropping due to the size of the cardinal B-splines.
            - b_groups: int. Split the convolution kernels (along the output channel axis) into groups that have their own set of centers (basis functions). If b_groups=out_channels, then each output channel is generated with a kernel that has its own basis consisting of n_basis functions.
            - bias: not implemented
        Output (of the generated layer):
            - output: torch.tensor, size=[B,Cout,X_1',X_2',...,X_d']. Here X_i' are the cropped/padded spatial dims.
        """
        # nn.Module init
        super(GConvGG, self).__init__()

        # The base settings
        self.group = group
        self.d = group.Rd.d
        self.Cin = in_channels
        self.Cout = out_channels
        self.size = kernel_size
        self.h_grid = h_grid
        self.h_grid_in = h_grid_in
        self.xMax = kernel_size // 2  # Specifies the maximum allowed pixel offset
        self.N = n_basis
        # The convolution settings
        self.padding = padding
        self.b_padding = b_padding
        self.stride = stride
        self.b_groups = b_groups
        self.b_groups_sigmas = b_groups_sigmas
        # The B-spline settings
        self.n = b_order
        self.s = b_scale
        self.adaptive_basis = adaptive_basis
        self.virtual_kernel_size = virtual_kernel_size
        self.dilation = dilation
        self.h_crop = h_crop

        if bias:
            # Random (normal dist) init
            self.bias = torch.nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

        # The indices (random if not specified, otherwise on a regular grid > dense kernel)
        x_centers, h_centers, self.N, self.x_min, self.x_max = x_h_centers_init(self.group, self.size,
                                                                                          self.h_grid, self.N,
                                                                                          b_groups=b_groups)

        # If also optimizing over the centers:
        if self.adaptive_basis:
            self.x_centers = torch.nn.Parameter(x_centers.float())
            self.h_centers = torch.nn.Parameter(h_centers.float())
        else:
            self.x_centers = x_centers
            self.h_centers = h_centers.type(torch.cuda.FloatTensor)  #(Observation) It's not neccessary. It's handled by .device()

        # Kaiming init
        self.weights = torch.nn.Parameter(torch.Tensor(self.Cout, self.Cin, self.h_grid.N, self.size))  # [N_out, N_h, N_in, N_h, X, Y] is the standard form in torch.  # TODO: H or H_in?
        # Initialization parameters
        wscale = math.sqrt(2.)  # This makes the initialization equal to that of He et al
        self._reset_parameters(wscale=wscale)

        if self.virtual_kernel_size is None:
            max_scaling = torch.max(self.group.H.scaling(h_grid.grid))
            max_center = torch.max(torch.abs(self.x_centers))
            self.virtual_kernel_size = int(torch.round(max_scaling * max_center)) * 2 + 1

    def forward(self, input):
        output = gconv_G_G(input, self.weights, self.x_centers, self.h_centers, self.virtual_kernel_size, self.group, self.h_grid, self.h_grid_in, self.n, self.s, self.stride, self.padding,
                       self.b_padding, self.b_groups, self.bias, self.dilation, self.h_crop)
        return output

    def _reset_parameters(self, wscale):
        n = self.Cin
        k = self.size ** self.d
        n *= k
        stdv = wscale * (1. / math.sqrt(n))
        self.stdv = stdv
        self.weights.data.uniform_(-stdv, stdv)


def x_centers_init( d, size , N = None , b_groups = 1, integer = False):
    # The min max range of the kernel
    x_min = (size // 2) - size + 1
    x_max = (size // 2)
    if N is None:   # TODO: Are these operations necessary?
        # Then construct a regular grid
        grid_base = torch.arange(x_min, x_max + 1).repeat([size] * (d - 1) + [1]).transpose(0, -1)
        grid = torch.stack([grid_base.transpose(0, dim) for dim in range(d)], -1)
        grid_flat = grid.reshape([-1, d])
        N = grid_flat.shape[0]
        x_centers = torch.cat([grid_flat]*b_groups, 0).reshape(-1, d)
    else:
        # Random sample points (uniform)
        if integer:
            # Only integer grid points can be sampled
            assert N <= size ** d, "Error: \"n_basis\" should be at most kernel_size^d to be able to initialize unique index sampling locations."
            grid_base = torch.arange(x_min, x_max + 1).repeat([size] * (d - 1) + [1]).transpose(0, -1)
            grid = torch.stack([grid_base.transpose(0, dim) for dim in range(d)], -1)
            grid_flat = grid.reshape([-1, d])
            x_centers = torch.cat([grid_flat[torch.randperm(grid_flat.shape[0])][:N] for bgroup in range(b_groups)],
                                  0).reshape(b_groups * N, d)
        else:
            # Any float within the min max range is OK
            x_centers = x_min + torch.rand(b_groups * N, d) * (x_max - x_min)
    return x_centers, N, x_min, x_max


def x_h_centers_init( group, size , h_grid, N = None , b_groups = 1 ):
    d = group.Rd.d
    # The min max range of the kernel
    x_min = (size // 2) - size + 1
    x_max = (size // 2)
    if N is None:
        # Then construct a regular grid
        x_grid_base = torch.arange(x_min, x_max + 1).repeat([size] * (d - 1) + [1]).transpose(0, -1)
        x_grid = torch.stack([x_grid_base.transpose(0, dim) for dim in range(d)], -1)
        x_grid_flat = x_grid.reshape([-1, d])
        # The h_grid copied for each point in the x_grid:
        h_grid_base = h_grid.grid
        h_grid_flat = torch.cat([torch.cat([h] * x_grid_flat.shape[0], 0) for h in h_grid_base], 0)  # [h1,h1,...,h2,h2
        # Copy for every h in h_grid:
        x_grid_flat = torch.cat([x_grid_flat] * h_grid.N, 0)  # ((x1,h1),(x2,h1),...,(x1,h2),(x2,h2),...)
        N = x_grid_flat.shape[0]
        # The centers:
        x_centers = torch.cat([x_grid_flat] * b_groups, 0).reshape(-1, d)
        h_centers = h_grid_flat
    else:
        # Any float within the min max range is OK
        x_centers = x_min + torch.rand(b_groups * N, d) * (x_max - x_min)
        h_centers = torch.rand(b_groups * N, d) * 2*np.pi
    return x_centers, h_centers, N, x_min, x_max


def weights_init( N , Cout, Cin ):
    weights = torch.randn(N, Cout, Cin) * np.sqrt(2.0 / (Cin * N))
    return weights