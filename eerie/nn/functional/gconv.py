import torch
from eerie.nn.functional.bconv import bconv
from eerie.bsplines.utils import B_supp
from eerie.bsplines.b_1d import B
import eerie

def gconv_Rd_G(
        input,
        weights,
        centers,
        size,
        group,
        h_grid,
        n=2,
        s=1.,
        stride=1,
        padding=0,
        b_padding=True,
        b_groups=1,
        bias=None,
        dilation=1):
    """ Performs d-dimensional convolution with B-spline convolution kernels.
    Args:
        - input: torch.tensor, shape=[B,Cin,X,Y,...].
        - weights: torch.tensor, shape=[N,Cout,Cin]. Here N is the number of non-zero weights in each kernel.
        - centers: torch.tensor, dtype=int, size=[N,d]. Here d is the spatial dimension of input. NOTE: The centers are relative to the center of the kernel and can thus be negative, but should be smaller than |center[i]|<size//2.
        - size: int. This is the virtual kernel size (determines the cropping).
    Args (optional):
        - n: int. The order of the B-spline.
        - s: float. The scale of each shifted cardinal B-spline.
        - stride: int. not implemented...
        - padding: int. Integer that specifies the amount of spatial padding on each side.
        - b_padding: boolean. Whether or not to automatically correct for cropping due to the size of the cardinal B-splines.
        - b_groups: int. Split the convolution kernels (along the output channel axis) into groups that have their own set of centers (basis functions). If b_groups=Cout, then each output channel is generated with a kernel that has its own basis consisting of n_basis functions.
    Output:
        - output: torch.tensor, size=[B,Cout,X',Y',...]. Here X',Y',... are the cropped/padded spatial dims.
    """
    # Check if scale is indeed a float (check needs to be done otherwise
    # we get unwanted casting to integers of the centers)
    assert isinstance(s, float), "The specified argument \"s\" should be a float."

    output = efficient_1Dspline_conv_R1G(input, weights, group, h_grid, 2, dilation)

    # Add bias if provided
    if bias is not None: #TODO: Not done yet
        pass #output += bias.reshape([1, Cout] + [1] * group.G.d)

    # Return the output
    return output

def efficient_1Dspline_conv_R1G(input, weights, group, h_grid, order_spline, dilation):
    # Method:
    # For each scale s, perform in parallel:
    # 1. Convolve input with B-spline (of scale s).
    # 2. Convolve with dilated convolutions (with dilation proportional to s).
    # 3. Concatenate responses (put away unneccessary padding).
    # TODO: Need to implement a similar (less-efficient version) for dilations < 1. It can be done by first computing base*weights n' performing usual convolution.

    output_cat = []
    for scale in h_grid.grid:
    # Check if scale is indeed a float (check needs to be done otherwise
    # we get unwanted casting to integers of the centers)
        if type(scale) is not torch.Tensor:
            assert isinstance(scale, float), "The specified argument \"s\" should be a float."

        # The cardinal b-spline # TODO put somewhere else, maybe in the definition of the layer.
        Bfunc = B(order_spline)
        _, xMax, brange = eerie.bsplines.utils.B_supp_grid_2(n=order_spline, s=scale, intsupp=True, device=input.device) # Compute the spline as well as the ammount of padding required.

        # Get values of cardinal spline on grid
        b_spline_on_grid = Bfunc( brange / scale )  # TODO why divided by s? / Can be computed offline

        # Convolve with the cardinal B-spline
        # Calculate convolution parameters ( With cardinal B-spline)
        # ----------------------------------------
        # padding: The required padding equals int(b_spline_on_grid.shape[0] + (scale/2)) and is a function of the scale.
        padding = int(b_spline_on_grid.shape[0] + (scale/2) * (weights.shape[-1] - 4)) # TODO * dilation

        N_b = input.shape[0]
        N_in = input.shape[1]
        # ----------------------------------------
        output = torch.conv1d(input=input.reshape(N_b * N_in, 1, input.shape[-1]), weight=b_spline_on_grid.view(1, 1, -1), bias=None, stride=1, padding=padding, dilation=1, groups=1)
        output = output.reshape(N_b, N_in, -1)
        # Convolve with weights
        # Calculate convolution parameters ( With weights)
        # ----------------------------------------
        # dilation: depends on scale
        dilation = int(scale)   # Equivalent to H.left_action_on_Rd
        # ----------------------------------------
        output = float(1/group.H.det(scale)) * torch.conv1d(input=output, weight=weights, bias=None, stride=1, padding=0, dilation=dilation, groups=1)
        #  The spatial dimension of the output is equal to that of the input.
        output_cat.append(output)

    # Concatenate all scales
    return torch.stack(output_cat, dim=2)


def gconv_G_G(
        input,
        weights,
        x_centers,
        h_centers,
        size,
        group,
        h_grid,
        h_grid_in=None,
        n=2,
        s=1.,
        stride=1,
        padding=0,
        b_padding=True,
        b_groups=1,
        bias=None,
        dilation=1,
        h_crop=False):
    """ Performs d-dimensional convolution with B-spline convolution kernels.
    Args:
        - input: torch.tensor, shape=[B,Cin,X,Y,...].
        - weights: torch.tensor, shape=[N,Cout,Cin]. Here N is the number of non-zero weights in each kernel.
        - centers: torch.tensor, dtype=int, size=[N,Rd.d+H.d]. Here Rd.d and H.d are the spatial and subgroup H dimensions, which together form the dimension of the input. NOTE: The centers are relative to the center of the kernel and can thus be negative, but should be smaller than |center[i]|<size//2.
        - size: int. This is the virtual kernel size (determines the cropping).
    Args (optional):
        - n: int. The order of the B-spline.
        - s: float. The scale of each shifted cardinal B-spline.
        - stride: int. not implemented...
        - padding: int. Integer that specifies the amount of spatial padding on each side.
        - b_padding: boolean. Whether or not to automatically correct for cropping due to the size of the cardinal B-splines.
        - b_groups: int. Split the convolution kernels (along the output channel axis) into groups that have their own set of centers (basis functions). If b_groups=Cout, then each output channel is generated with a kernel that has its own basis consisting of n_basis functions.
    Output:
        - output: torch.tensor, size=[B,Cout,X',Y',...]. Here X',Y',... are the cropped/padded spatial dims.
    """
    # Check if scale is indeed a float (check needs to be done otherwise
    # we get unwanted casting to integers of the centers)
    assert isinstance(s, float), "The specified argument \"s\" should be a float."

    if h_grid_in is None:
        h_grid_in = h_grid

    output = efficient_1Dspline_conv_GG_locscalefilters(input, weights, group, h_grid, h_grid_in, 2, dilation,h_crop=h_crop)

    # Add bias if provided  # TODO
    if bias is not None: # TODO implement bias
        pass #output += bias.reshape([1, Cout] + [1] * group.G.d)

    # Return the output
    return output


def merge_channel_and_h_axes(input):
    # input has shape [B,C,X,Y,...,H]
    return torch.cat([input[...,hindex] for hindex in range(input.shape[-1])],1)


def efficient_1Dspline_conv_GG_locscalefilters(input, weights, group, h_grid, h_grid_in, order_spline, dilation, h_crop = False):
    # Method:
    # For each scale s in N_h, perform in parallel:
    # 1. Vectorize input into input of form [B, N_out, N_h_in * N_in]
    # 2. Convolve input with B-spline (of scale s).
    # 3. Convolve with dilated convolutions (with dilation proportional to s).
    # 4. Concatenate responses (put away unneccessary padding).
    # TODO: Need to implement a similar (less-efficient version) for dilations < 1. It can be done by first computing base*weights n' performing usual convolution.

    output_cat = []

    if h_crop:
        crop_factor = h_grid.N - 1
    else:
        crop_factor = 0
    for in_scale in range(1, input.shape[-2] + 1 - crop_factor):     #[N_h_in]

        # keep track of where we are
        h_grid_count = in_scale - 1 # Use to keep track of which output scales have been calculated. (it's used to shrink sizes of input/weights) as some become zero when moving accross scales.

        # we want to avoid unwanted casting to integers.
        in_scale = float(2 ** (in_scale - 1))

        Bfunc = B(order_spline)
        _, xMax, brange = eerie.bsplines.utils.B_supp_grid_2(n=order_spline, s=in_scale, intsupp=True, device=input.device) # Scales start from 1 to N_h

        # Get values of cardinal spline on grid
        b_spline_on_grid = Bfunc(brange / (in_scale))  # TODO why divided by s? / Can be computed offline

        # Convolve with the cardinal B-spline
        # Calculate convolution parameters (with cardinal B-spline)
        # ----------------------------------------
        # padding: The required padding equals int(b_spline_on_grid.shape[0] + (scale/2)) and is a function of the scale.
        padding = int(dilation * in_scale * (weights.shape[-1] // 2) + b_spline_on_grid.shape[0] // 2)

        # input needs to be reshaped as [B * N_in * N_h_in, 1, X]
        N_b = input.shape[0]
        N_in = input.shape[1]
        N_h_in = input.shape[2]
        # ----------------------------------------  # The h_grid_count: incorporates the fact that the highest resolutions are not considered for coarser scales.
        output = torch.conv1d(input=input[:, :, h_grid_count:h_grid_count+h_grid.N, :].reshape(-1, 1, input.shape[-1])
                              if (h_grid_count + h_grid.N) < input.shape[-2] else
                              input[:, :, h_grid_count:, :].reshape(-1, 1, input.shape[-1]),
                              weight=b_spline_on_grid.view(1, 1, -1), bias=None, stride=1, padding=padding, dilation=1, groups=1)
        # Reshape to input-like form
        shape_inter = h_grid.N if (h_grid_count + h_grid.N) < input.shape[-2] else (N_h_in - h_grid_count)
        output = output.reshape(N_b, N_in, shape_inter, -1)

        # Convolve with weights
        # Calculate convolution parameters (with weights)
        # ----------------------------------------
        # dilation: depends on scale
        dil = int(in_scale) * dilation # Equivalent to H.left_action_on_Rd
        # ---------------------------------------- # The h_grid_count: incorporates the fact that the coarsest part of the kernel are not considered for coarser scales.
        output = output.reshape(N_b, N_in * shape_inter, -1)
        weight = weights[:, :, :, :].reshape(weights.shape[0], -1, weights.shape[-1]) if (in_scale + h_grid.N) < input.shape[-2] \
                else weights[:, :, :shape_inter, :].reshape(weights.shape[0], -1, weights.shape[-1])
        output = float(1 / group.H.det(in_scale) ) * torch.conv1d(input=output, weight=weight, bias=None, stride=1, padding=0, dilation=dil, groups=1)

        #  The spatial dimension of the output is equal to that of the input.
        output_cat.append(output)

    # Concatenate all scales
    return torch.stack(output_cat, dim=2)
