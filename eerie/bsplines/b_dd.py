# -*- coding: utf-8 -*-
"""
Implementation of d-Dimensional B-splines

File created Wed 11 Dec 2019 11:16:51
@author: EJ Bekkers, Institute for Informatics, University of Amsterdam, The Netherlands

"""

import torch as torch
import math as m
from eerie.bsplines.b_1d import B as B_R1


## The d-dimensional B-spline
def B(n):
    """ Returns a d-dimensional B-spline basis function of degree "n" (centered
        around zero). 

        INPUT:
            - degree n, an integer

        OUTPUT:
            - func, a python function which takes as input a torch.Tensor whose last
              dimension encodes the coordinates. E.g. B(2)([0,0.5]) computes the
              value at coordinate [0,0.5] and B(2)([[0,0.5],[0.5,0.5]]) returns 
              the values at coordinates [0,0.5] and [0.5,0.5]. This is also the
              case for a 1D B-spline: B(2)([[0],[0.5]]) returns the values of the
              1D B-spline at coordinates 0 and 0.5.
    """
    def B_Rd(x):
        return torch.prod(B_R1(n)(x),-1)
    return B_Rd
