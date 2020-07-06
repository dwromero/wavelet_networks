# Class implementation of the scale-translation group for 1D data
import torch
import numpy as np

# TODO: implement action on Rd with origin... Class H

# Rules for setting up a group class:
# A group element is always stored as a 1D vector, even if the elements consist
# only of a scalar (in which case the element is a list of length 1). Here we 
# also assume that you can parameterize your group with a set of n parameters,
# with n the dimension of the group. The group elements are thus always lists of
# length n.
#
# This file requires the definition of the base/normal sub-group R^n and the 
# sub-group H. Together they will define G = R^d \rtimes H.
#
# In order to derive G (it's product and inverse) we need for the group H to be
# known the group product, inverse and left action on R^d.
#
# In the B-Spline networks we also need to know a distance between two elements
# g_1 and g_2 in H. This can be defined by computing the length (l2 norm) of the
# vector that brings your from element g_1 to g_2 via the exponential map. This 
# vector can be obtained by taking the logarithmic map of (g_1^{-1} g_2). Since
# the B-splines are symmetric we can cheat a bit and allow the distance to be
# negative, as long as true_dist(x,y) = abs(dist(x,y)). E.g. the Euclidean distance
# between two scalars is true_dist(x,y) = abs(x - y), but it saves us a tiny bit of
# of computation to omit the abs.
#
# Finally we need a way to sample the group. Therefore also a function "grid" is
# defined which samples the group as uniform as possible given a specified 
# number of elements N. Not all groups allow a uniform sampling given an 
# aribitrary N, in which case the sampling can be made approximately uniform by 
# maximizing the distance between all sampled elements in H (e.g. via a 
# repulsion model).

# The normal sub-group R^d:
# This is just the vector space R^d (translations) with the group product and inverse defined
# via the + and -.
class Rd(torch.nn.Module):  # <------------------------------------------------------------------------------------ CHANGE FOR NEW GROUPS
    def __init__(self):
        # nn.Module init
        super(Rd, self).__init__()
        self.name = 'R^1'         # Label for the group
        self.d = 1         # Dimension of the base manifold N=R^d
        # self.e = torch.tensor(np.array([0., 0.], dtype=np.float32))         # The identity element
        self.register_buffer('e',torch.from_numpy(np.array([0.], dtype=np.float32)))

# The sub-group H:
class H(torch.nn.Module):  # <------------------------------------------------------------------------------------- CHANGE FOR NEW GROUPS
    def __init__(self):
        super(H, self).__init__()
        self.name = 'R+'   # Label for the group
        self.d = 1  # Dimension of the sub-group H# Each element consists of 1 parameter
        self.register_buffer('e', torch.from_numpy(np.array([1.], dtype=np.float32)) )
        # self.e = torch.tensor(np.array([0.], dtype=np.float32))        # The identify element

    # Essential definitions of the group
    # Group product
    def prod(self, h_1, h_2):  # <-------------------------------------------------------- CHANGE FOR NEW GROUP IMPLEMENTATION
        return h_1 * h_2

    # Group inverse
    def inv(self, h):  # <---------------------------------------------------------------- CHANGE FOR NEW GROUP IMPLEMENTATION
        return 1 / h

    # Essential for computing the distance between two group elements
    # Logarithmic map
    def log(self, h):  # <---------------------------------------------------------------- CHANGE FOR NEW GROUP IMPLEMENTATION
        return torch.log(h)

    def exp(self, c):  # <---------------------------------------------------------------- CHANGE FOR NEW GROUP IMPLEMENTATION
        return torch.exp(c)

    # Distance between two group elements
    def dist(self, h_1, h_2):  # <-------------------------------------------------------- CHANGE FOR NEW GROUP IMPLEMENTATION
        # The logarithmic distance ||log(inv(h1).h2)||
        dist = (self.log(self.prod(self.inv(h_1), h_2)))[..., 0]  # Since each h is a list of length 1 we can do [...,0]
        return dist

    # Essential for constructing the group G = R^d \rtimes H
    # Define how H acts transitively on R^d
    def left_action_on_Rd(self, h, xx, xx_0 = None):  # <---------------------------------------------- CHANGE FOR NEW GROUP IMPLEMENTATION
        s = h[0]
        x = xx.squeeze(-1)
        if xx_0 is None:
            x_new = x * s
        else:
            x_new = (x - xx_0[0]) * s
        # Reformat c
        xx_new = torch.stack([x_new], axis=-1)  # TODO: replace with .unsqueeze(-1)
        # Return the result
        return xx_new

    # Essential in the group convolutions
    # Define the determinant (of the matrix representation) of the group element
    def det(self, h):  # <---------------------------------------------------------------- CHANGE FOR NEW GROUP IMPLEMENTATION
        return h

    def scaling(self, h):
        return h

# Grid class
# The local grid class does not needs to be re-implemented and can be reused for every Lie group H
class h_grid_local(torch.nn.Module):  # A local grid based on the exponential map
    # Should a least contain:
    # N     - specifies the number of grid points
    # scale - the (approximate) distance between points in the Lie algebra, this will be used to scale the B-splines
    # grid  - the actual grid
    # args  - such that we always know how the grid was constructed
    # Construct the grid
    def __init__(self, N, scale):
        super(h_grid_local, self).__init__()
        # This rembembers the arguments used to construct the grid (this is to make it a bit more future proof, you
        # may want to define a grid using specific parameters and later in the code construct a similar grid with
        # the same parameters, but with N changed for example)
        self.args = locals().copy()
        self.args.pop('self')
        # Store N
        self.N = N
        # Define the scale (the spacing between points)
        # dc should be a list of length H.d. Let's turn it into a numpy array:
        scale_np = np.array(scale,dtype=np.float32)
        self.register_buffer('scale',torch.from_numpy(scale_np))

        # Generate the grid

        # Create an array of uniformly spaced exp. coordinates (step size is 1):
        # The indices always include 0. When N = odd, the grid is symmetric. E.g. N=3 -> [-1,0,1].
        # When N = even the grid is moved a bit to the right. E.g. N=2 -> [0,1], N=3 -> [-1,0,1,2]
        grid_start = -((N - 1) // 2)
        c_index_array = np.moveaxis(np.mgrid[tuple([slice(grid_start, grid_start + N)] * H().d)], 0, -1).astype(
            np.float32)
        # Scale the grid with dc
        c_array = scale_np * c_index_array
        # Flatten it to a list of exp coordinates
        c_list = np.reshape(c_array, [-1, H().d])
        # Turn it into group elements via the exponential map
        h_list = H().exp(c_list)
        # Save the generated grid
        self.register_buffer('grid',torch.from_numpy(h_list))

# The global grid is a specialized class and should at least contain N
class h_grid_global(torch.nn.Module):  # <--------------------------------------------------------- CHANGE FOR NEW GROUP IMPLEMENTATION
    # Should a least contain:
    # N     - specifies the number of grid points
    # scale - the (approximate) distance between points (in the Lie algebra), it is used to scale the B-splines
    # grid  - the actual grid
    # args  - such that we always know how the grid was construted
    # Construct the grid
    def __init__(self, N, scale_range):
        super(h_grid_global, self).__init__()
        # This rembembers the arguments used to construct the grid (this is to make it a bit more future proof, you
        # may want to define a grid using specific parameters and later in the code construct a similar grid with
        # the same parameters, but with N changed for example)
        self.H = H()
        self.args = locals().copy()
        self.args.pop('self')
        # Store N
        self.N = N
        # Define the scale (the spacing between points)
        if self.N > 1:
            scale_np = np.array(np.log(scale_range) / (N - 1), dtype=np.float32)
        else:
            scale_np = np.array(1., dtype=np.float32)
        self.register_buffer('scale',torch.from_numpy( scale_np ) )
        # Generate the grid
        c_index_array = np.moveaxis(np.mgrid[tuple([slice(0, self.N)] * self.H.d)], 0, -1).astype(np.float32)
        # Scale the grid with dc
        c_array = scale_np * c_index_array
        # Flatten it to a list of exp coordinates as a tensorflow constant
        c_list = torch.from_numpy(np.reshape(c_array, [-1, self.H.d]))
        # Turn it into group elements via the exponential map
        h_list = self.H.exp(c_list)
        # Save the generated grid
        self.register_buffer('grid',h_list)


# The derived group G = R^d \rtimes H.
# The above translation group and the defined group H together define the group G
# The following is automatically constructed and should not be changed unless
# you may have some speed improvements, or you may want to add some functions such
# as the logarithmic and exponential map.
# A group element in G should always be a vector of length Rd.d + H.d
class G(torch.nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.Rd = Rd()
        self.H = H()
        self.name = 'SE(2)'   # Label for the group
        self.d = self.Rd.d + self.H.d  # Dimension of the sub-group H# Each element consists of 1 parameter
        self.e = torch.cat([self.Rd.e, self.H.e], axis=-1)       # The identify element

    # Function for splitting a group element g in G in to its xx and h component
    def xx_h(self, g):
        Rd = self.Rd
        xx = g[..., 0:Rd.d]
        h = g[..., Rd.d:]
        return xx, h

    # Function that returns the classes for R^n and H
    def Rd_H(self):
        return self.Rd, self.H

    # The derived group product
    def prod(self, g_1, g_2):  # Input g_1 is a single group element, input g_2 can be an array of group elements
        # (xx_1, h_1)
        Rd = self.Rd
        H = self.H

        xx_1 = g_1[0:Rd.d]
        h_1 = g_1[Rd.d:]
        # (xx_2, h_2)
        xx_2 = g_2[..., 0:Rd.d]
        h_2 = g_2[..., Rd.d:]
        # (xx_new, h_new)
        xx_new = H.left_action_on_Rd(h_1, xx_2) + xx_1
        h_new = H.prod(h_1, h_2)
        g_new = torch.cat([xx_new, h_new], axis=-1)
        # Return the result
        return g_new

    # The derived group inverse
    def inv(self, g):
        # g = (xx, h)
        Rd = self.Rd
        H = self.H

        xx = g[0:Rd.d]
        h = g[Rd.d:]
        # Compute the inverse
        h_inv = H.inv(h)
        xx_inv = H.left_action_on_Rd(H.inv(h), -xx)
        # Reformat g3
        g_inv = torch.cat([xx_inv, h_inv], axis=-1)
        # Return the result
        return g_inv
