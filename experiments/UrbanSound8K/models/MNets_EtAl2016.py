# torch
import torch
import torch.nn as nn
# built-in
import functools
# project
import eerie

class R_M3(torch.nn.Module):
    def __init__(self, use_bias=False):
        super(R_M3, self).__init__()
        # Parameters of the model
        use_bias = False
        eps = 2e-5
        n_channels = 256
        n_classes = 10

        # Conv Layers
        self.c1 = torch.nn.Conv1d(in_channels=1,          out_channels=n_channels, kernel_size=80, stride=4, padding=(80 // 2), dilation=1, bias=use_bias)
        self.c2 = torch.nn.Conv1d(in_channels=n_channels, out_channels=n_channels, kernel_size=3,  stride=1, padding=(3 // 2),  dilation=1, bias=use_bias)
        self.c3 = torch.nn.Conv1d(in_channels=n_channels, out_channels=n_classes,  kernel_size=1,  stride=1, padding=0,         dilation=1, bias=use_bias)
        # BatchNorm Layers
        self.bn1 = torch.nn.BatchNorm1d(num_features=n_channels, eps=eps)
        self.bn2 = torch.nn.BatchNorm1d(num_features=n_channels, eps=eps)

        # Pooling
        self.pool = torch.max_pool1d

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                m.weight.data.normal_(0, torch.prod(torch.Tensor(list(m.weight.shape)[1:]))**(-1/2))
                if use_bias: m.bias.data.fill_(0.0)

    def forward(self, x):
        out = torch.relu(self.bn1(self.c1(x)))
        out = self.pool(out, kernel_size=4, stride=4, padding=0)
        out = torch.relu(self.bn2(self.c2(out)))
        out = self.pool(out, kernel_size=4, stride=4, padding=0)
        # Global pooling
        out = torch.mean(out, dim=-1, keepdim=True) # pool over the time axis
        out = self.c3(out)
        out = out.view(out.size(0), 10)
        return out


class R_M5(torch.nn.Module):
    def __init__(self, use_bias=False):
        super(R_M5, self).__init__()
        # Parameters of the model
        use_bias = False
        eps = 2e-5
        n_channels = 128
        n_classes = 10

        # Conv Layers
        self.c1 = torch.nn.Conv1d(in_channels=1,              out_channels=n_channels,     kernel_size=80, stride=4, padding=(80 // 2), dilation=1, bias=use_bias)
        self.c2 = torch.nn.Conv1d(in_channels=n_channels,     out_channels=n_channels,     kernel_size=3,  stride=1, padding=(3 // 2),  dilation=1, bias=use_bias)
        self.c3 = torch.nn.Conv1d(in_channels=n_channels,     out_channels=n_channels * 2, kernel_size=3, stride=1,  padding=(3 // 2),  dilation=1, bias=use_bias)
        self.c4 = torch.nn.Conv1d(in_channels=n_channels * 2, out_channels=n_channels * 4, kernel_size=3, stride=1,  padding=(3 // 2),  dilation=1, bias=use_bias)
        self.c5 = torch.nn.Conv1d(in_channels=n_channels * 4, out_channels=n_classes,      kernel_size=1,  stride=1, padding=0,         dilation=1, bias=use_bias)
        # BatchNorm Layers
        self.bn1 = torch.nn.BatchNorm1d(num_features=n_channels,     eps=eps)
        self.bn2 = torch.nn.BatchNorm1d(num_features=n_channels,     eps=eps)
        self.bn3 = torch.nn.BatchNorm1d(num_features=n_channels * 2, eps=eps)
        self.bn4 = torch.nn.BatchNorm1d(num_features=n_channels * 4, eps=eps)

        # Pooling
        self.pool = torch.max_pool1d

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                m.weight.data.normal_(0, torch.prod(torch.Tensor(list(m.weight.shape)[1:]))**(-1/2))
                if use_bias: m.bias.data.fill_(0.0)

    def forward(self, x):
        out = torch.relu(self.bn1(self.c1(x)))
        out = self.pool(out, kernel_size=4, stride=4, padding=0)
        out = torch.relu(self.bn2(self.c2(out)))
        out = self.pool(out, kernel_size=4, stride=4, padding=0)
        out = torch.relu(self.bn3(self.c3(out)))
        out = self.pool(out, kernel_size=4, stride=4, padding=0)
        out = torch.relu(self.bn4(self.c4(out)))
        out = self.pool(out, kernel_size=4, stride=4, padding=0)
        # Global pooling
        out = torch.mean(out, dim=-1, keepdim=True) # pool over the time axis
        out = self.c5(out)
        out = out.view(out.size(0), 10)
        return out


class R_M11(torch.nn.Module):
    def __init__(self, use_bias=False):
        super(R_M11, self).__init__()
        # Parameters of the model
        use_bias = False
        eps = 2e-5
        n_channels = 64
        n_classes = 10

        # Conv Layers
        self.c1 = torch.nn.Conv1d(in_channels=1,              out_channels=n_channels,     kernel_size=80, stride=4, padding=(80 // 2), dilation=1, bias=use_bias)
        self.c2 = torch.nn.Conv1d(in_channels=n_channels,     out_channels=n_channels,     kernel_size=3,  stride=1, padding=(3 // 2),  dilation=1, bias=use_bias)
        self.c3 = torch.nn.Conv1d(in_channels=n_channels,     out_channels=n_channels,     kernel_size=3,  stride=1, padding=(3 // 2),  dilation=1, bias=use_bias)
        self.c4 = torch.nn.Conv1d(in_channels=n_channels,     out_channels=n_channels * 2, kernel_size=3, stride=1,  padding=(3 // 2),  dilation=1, bias=use_bias)
        self.c5 = torch.nn.Conv1d(in_channels=n_channels * 2, out_channels=n_channels * 2, kernel_size=3, stride=1,  padding=(3 // 2), dilation=1, bias=use_bias)
        self.c6 = torch.nn.Conv1d(in_channels=n_channels * 2, out_channels=n_channels * 4, kernel_size=3, stride=1,  padding=(3 // 2),  dilation=1, bias=use_bias)
        self.c7 = torch.nn.Conv1d(in_channels=n_channels * 4, out_channels=n_channels * 4, kernel_size=3, stride=1,  padding=(3 // 2), dilation=1, bias=use_bias)
        self.c8 = torch.nn.Conv1d(in_channels=n_channels * 4, out_channels=n_channels * 4, kernel_size=3, stride=1,  padding=(3 // 2), dilation=1, bias=use_bias)
        self.c9 = torch.nn.Conv1d(in_channels=n_channels * 4, out_channels=n_channels * 8, kernel_size=3, stride=1,  padding=(3 // 2), dilation=1, bias=use_bias)
        self.c10 = torch.nn.Conv1d(in_channels=n_channels * 8, out_channels=n_channels * 8, kernel_size=3, stride=1, padding=(3 // 2), dilation=1, bias=use_bias)
        self.c11 = torch.nn.Conv1d(in_channels=n_channels * 8, out_channels=n_classes,      kernel_size=1,  stride=1, padding=0,         dilation=1, bias=use_bias)
        # BatchNorm Layers
        self.bn1 = torch.nn.BatchNorm1d(num_features=n_channels, eps=eps)
        self.bn2 = torch.nn.BatchNorm1d(num_features=n_channels, eps=eps)
        self.bn3 = torch.nn.BatchNorm1d(num_features=n_channels, eps=eps)
        self.bn4 = torch.nn.BatchNorm1d(num_features=n_channels * 2, eps=eps)
        self.bn5 = torch.nn.BatchNorm1d(num_features=n_channels * 2, eps=eps)
        self.bn6 = torch.nn.BatchNorm1d(num_features=n_channels * 4, eps=eps)
        self.bn7 = torch.nn.BatchNorm1d(num_features=n_channels * 4, eps=eps)
        self.bn8 = torch.nn.BatchNorm1d(num_features=n_channels * 4, eps=eps)
        self.bn9 = torch.nn.BatchNorm1d(num_features=n_channels * 8, eps=eps)
        self.bn10 = torch.nn.BatchNorm1d(num_features=n_channels * 8, eps=eps)

        # Pooling
        self.pool = torch.max_pool1d

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                m.weight.data.normal_(0, torch.prod(torch.Tensor(list(m.weight.shape)[1:]))**(-1/2))
                if use_bias: m.bias.data.fill_(0.0)

    def forward(self, x):
        out = torch.relu(self.bn1(self.c1(x)))
        out = self.pool(out, kernel_size=4, stride=4, padding=0)
        out = torch.relu(self.bn2(self.c2(out)))
        out = torch.relu(self.bn3(self.c3(out)))
        out = self.pool(out, kernel_size=4, stride=4, padding=0)
        out = torch.relu(self.bn4(self.c4(out)))
        out = torch.relu(self.bn5(self.c5(out)))
        out = self.pool(out, kernel_size=4, stride=4, padding=0)
        out = torch.relu(self.bn6(self.c6(out)))
        out = torch.relu(self.bn7(self.c7(out)))
        out = torch.relu(self.bn8(self.c8(out)))
        out = self.pool(out, kernel_size=4, stride=4, padding=0)
        out = torch.relu(self.bn9(self.c9(out)))
        out = torch.relu(self.bn10(self.c10(out)))
        # Global pooling
        out = torch.mean(out, dim=-1, keepdim=True) # pool over the time axis
        out = self.c11(out)
        out = out.view(out.size(0), 10)
        return out


class R_M18(torch.nn.Module):
    def __init__(self, use_bias=False):
        super(R_M18, self).__init__()
        # Parameters of the model
        use_bias = False
        eps = 2e-5
        n_channels = 64
        n_classes = 10

        # Conv Layers
        self.c1 = torch.nn.Conv1d(in_channels=1,              out_channels=n_channels,     kernel_size=80, stride=4, padding=(80 // 2), dilation=1, bias=use_bias)
        self.c2 = torch.nn.Conv1d(in_channels=n_channels,     out_channels=n_channels,     kernel_size=3,  stride=1, padding=(3 // 2),  dilation=1, bias=use_bias)
        self.c3 = torch.nn.Conv1d(in_channels=n_channels,     out_channels=n_channels,     kernel_size=3,  stride=1, padding=(3 // 2),  dilation=1, bias=use_bias)
        self.c4 = torch.nn.Conv1d(in_channels=n_channels,     out_channels=n_channels,     kernel_size=3,  stride=1, padding=(3 // 2), dilation=1, bias=use_bias)
        self.c5 = torch.nn.Conv1d(in_channels=n_channels,     out_channels=n_channels,     kernel_size=3,  stride=1, padding=(3 // 2), dilation=1, bias=use_bias)
        self.c6 = torch.nn.Conv1d(in_channels=n_channels,     out_channels=n_channels * 2, kernel_size=3,  stride=1, padding=(3 // 2),  dilation=1, bias=use_bias)
        self.c7 = torch.nn.Conv1d(in_channels=n_channels * 2, out_channels=n_channels * 2, kernel_size=3,  stride=1, padding=(3 // 2), dilation=1, bias=use_bias)
        self.c8 = torch.nn.Conv1d(in_channels=n_channels * 2, out_channels=n_channels * 2, kernel_size=3,  stride=1, padding=(3 // 2), dilation=1, bias=use_bias)
        self.c9 = torch.nn.Conv1d(in_channels=n_channels * 2, out_channels=n_channels * 2, kernel_size=3,  stride=1, padding=(3 // 2), dilation=1, bias=use_bias)
        self.c10 = torch.nn.Conv1d(in_channels=n_channels * 2, out_channels=n_channels * 4, kernel_size=3, stride=1, padding=(3 // 2),  dilation=1, bias=use_bias)
        self.c11 = torch.nn.Conv1d(in_channels=n_channels * 4, out_channels=n_channels * 4, kernel_size=3, stride=1, padding=(3 // 2), dilation=1, bias=use_bias)
        self.c12 = torch.nn.Conv1d(in_channels=n_channels * 4, out_channels=n_channels * 4, kernel_size=3, stride=1, padding=(3 // 2), dilation=1, bias=use_bias)
        self.c13 = torch.nn.Conv1d(in_channels=n_channels * 4, out_channels=n_channels * 4, kernel_size=3, stride=1, padding=(3 // 2), dilation=1, bias=use_bias)
        self.c14 = torch.nn.Conv1d(in_channels=n_channels * 4, out_channels=n_channels * 8, kernel_size=3, stride=1, padding=(3 // 2), dilation=1, bias=use_bias)
        self.c15 = torch.nn.Conv1d(in_channels=n_channels * 8, out_channels=n_channels * 8, kernel_size=3, stride=1, padding=(3 // 2), dilation=1, bias=use_bias)
        self.c16 = torch.nn.Conv1d(in_channels=n_channels * 8, out_channels=n_channels * 8, kernel_size=3, stride=1, padding=(3 // 2), dilation=1, bias=use_bias)
        self.c17 = torch.nn.Conv1d(in_channels=n_channels * 8, out_channels=n_channels * 8, kernel_size=3, stride=1, padding=(3 // 2), dilation=1, bias=use_bias)
        self.c18 = torch.nn.Conv1d(in_channels=n_channels * 8, out_channels=n_classes,      kernel_size=1, stride=1, padding=0,         dilation=1, bias=use_bias)

        # BatchNorm Layers
        self.bn1 = torch.nn.BatchNorm1d(num_features=n_channels, eps=eps)
        self.bn2 = torch.nn.BatchNorm1d(num_features=n_channels, eps=eps)
        self.bn3 = torch.nn.BatchNorm1d(num_features=n_channels, eps=eps)
        self.bn4 = torch.nn.BatchNorm1d(num_features=n_channels, eps=eps)
        self.bn5 = torch.nn.BatchNorm1d(num_features=n_channels, eps=eps)
        self.bn6 = torch.nn.BatchNorm1d(num_features=n_channels * 2, eps=eps)
        self.bn7 = torch.nn.BatchNorm1d(num_features=n_channels * 2, eps=eps)
        self.bn8 = torch.nn.BatchNorm1d(num_features=n_channels * 2, eps=eps)
        self.bn9 = torch.nn.BatchNorm1d(num_features=n_channels * 2, eps=eps)
        self.bn10 = torch.nn.BatchNorm1d(num_features=n_channels * 4, eps=eps)
        self.bn11 = torch.nn.BatchNorm1d(num_features=n_channels * 4, eps=eps)
        self.bn12 = torch.nn.BatchNorm1d(num_features=n_channels * 4, eps=eps)
        self.bn13 = torch.nn.BatchNorm1d(num_features=n_channels * 4, eps=eps)
        self.bn14 = torch.nn.BatchNorm1d(num_features=n_channels * 8, eps=eps)
        self.bn15 = torch.nn.BatchNorm1d(num_features=n_channels * 8, eps=eps)
        self.bn16 = torch.nn.BatchNorm1d(num_features=n_channels * 8, eps=eps)
        self.bn17 = torch.nn.BatchNorm1d(num_features=n_channels * 8, eps=eps)

        # Pooling
        self.pool = torch.max_pool1d

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                m.weight.data.normal_(0, torch.prod(torch.Tensor(list(m.weight.shape)[1:]))**(-1/2))
                if use_bias: m.bias.data.fill_(0.0)

    def forward(self, x):
        out = torch.relu(self.bn1(self.c1(x)))
        out = self.pool(out, kernel_size=4, stride=4, padding=0)
        out = torch.relu(self.bn2(self.c2(out)))
        out = torch.relu(self.bn3(self.c3(out)))
        out = torch.relu(self.bn4(self.c4(out)))
        out = torch.relu(self.bn5(self.c5(out)))
        out = self.pool(out, kernel_size=4, stride=4, padding=0)
        out = torch.relu(self.bn6(self.c6(out)))
        out = torch.relu(self.bn7(self.c7(out)))
        out = torch.relu(self.bn8(self.c8(out)))
        out = torch.relu(self.bn9(self.c9(out)))
        out = self.pool(out, kernel_size=4, stride=4, padding=0)
        out = torch.relu(self.bn10(self.c10(out)))
        out = torch.relu(self.bn11(self.c11(out)))
        out = torch.relu(self.bn12(self.c12(out)))
        out = torch.relu(self.bn13(self.c13(out)))
        out = self.pool(out, kernel_size=4, stride=4, padding=0)
        out = torch.relu(self.bn14(self.c14(out)))
        out = torch.relu(self.bn15(self.c15(out)))
        out = torch.relu(self.bn16(self.c16(out)))
        out = torch.relu(self.bn17(self.c17(out)))
        # Global pooling
        out = torch.mean(out, dim=-1, keepdim=True) # pool over the time axis
        out = self.c18(out)
        out = out.view(out.size(0), 10)
        return out


class IdentityBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, use_bias, eps):
        super(IdentityBlock, self).__init__()

        # Conv Layers
        self.c1 = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                  padding=(kernel_size // 2), dilation=dilation, bias=use_bias)
        self.c2 = torch.nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                  padding=(kernel_size // 2), dilation=dilation, bias=use_bias)

        # BatchNorm Layers
        self.bn1 = torch.nn.BatchNorm1d(num_features=out_channels, eps=eps)
        self.bn2 = torch.nn.BatchNorm1d(num_features=out_channels, eps=eps)
        self.bn_out = torch.nn.BatchNorm1d(num_features=out_channels, eps=eps)

        # Additional params
        self.diff_size = (in_channels != out_channels)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                m.weight.data.normal_(0, torch.prod(torch.Tensor(list(m.weight.shape)[1:])) ** (-1 / 2))
                if use_bias: m.bias.data.fill_(0.0)

    def forward(self, x):
        out = torch.relu(self.bn1(self.c1(x)))
        out = self.bn2(self.c2(out))
        # shortcut
        if self.diff_size:
            out = out + x.repeat(1, 2, 1)
        else:
            out = out + x
        out = torch.relu(self.bn_out(out))
        return out


class R_M34res(torch.nn.Module):
    def __init__(self, use_bias=False):
        super(R_M34res, self).__init__()
        # Parameters of the model
        use_bias = False
        eps = 2e-5
        n_channels = 48
        n_classes = 10

        # Conv Layers
        self.c1 = torch.nn.Conv1d(in_channels=1,                        out_channels=n_channels,    kernel_size=80, stride=4, padding=(80 // 2), dilation=1, bias=use_bias)
        # ----
        first_block = []
        for i in range(3):
            first_block.append(IdentityBlock(in_channels=n_channels,    out_channels=n_channels,     kernel_size=3, stride=1, dilation=1, use_bias=use_bias, eps=eps))
        self.first_block = nn.Sequential(*first_block)
        # ----
        sec_block = []
        for i in range(4):
            b_channels = n_channels if i == 0 else n_channels * 2
            sec_block.append(IdentityBlock(in_channels=b_channels,      out_channels=n_channels * 2, kernel_size=3, stride=1, dilation=1, use_bias=use_bias, eps=eps))
        self.sec_block = nn.Sequential(*sec_block)
        # ----
        thrd_block = []
        for i in range(6):
            b_channels = n_channels * 2 if i == 0 else n_channels * 4
            thrd_block.append(IdentityBlock(in_channels=b_channels, out_channels=n_channels * 4, kernel_size=3, stride=1, dilation=1, use_bias=use_bias, eps=eps))
        self.thrd_block = nn.Sequential(*thrd_block)
        # ----
        frth_block = []
        for i in range(3):
            b_channels = n_channels * 4 if i == 0 else n_channels * 8
            frth_block.append(IdentityBlock(in_channels=b_channels, out_channels=n_channels * 8, kernel_size=3, stride=1, dilation=1, use_bias=use_bias, eps=eps))
        self.frth_block = nn.Sequential(*frth_block)
        # ----
        self.c_out = torch.nn.Conv1d(in_channels=n_channels * 8,        out_channels=n_classes,      kernel_size=1, stride=1, padding=0, dilation=1, bias=use_bias)

        # BatchNorm Layers
        self.bn1 = torch.nn.BatchNorm1d(num_features=n_channels, eps=eps)

        # Pooling
        self.pool = torch.max_pool1d

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                m.weight.data.normal_(0, torch.prod(torch.Tensor(list(m.weight.shape)[1:]))**(-1/2))
                if use_bias: m.bias.data.fill_(0.0)

    def forward(self, x):
        out = torch.relu(self.bn1(self.c1(x)))
        out = self.pool(out, kernel_size=4, stride=4, padding=0)
        out = self.first_block(out)
        out = self.pool(out, kernel_size=4, stride=4, padding=0)
        out = self.sec_block(out)
        out = self.pool(out, kernel_size=4, stride=4, padding=0)
        out = self.thrd_block(out)
        out = self.pool(out, kernel_size=4, stride=4, padding=0)
        out = self.frth_block(out)
        # Global pooling
        out = torch.mean(out, dim=-1, keepdim=True) # pool over the time axis
        out = self.c_out(out)
        out = out.view(out.size(0), 10)
        return out

# Dilation-translation equivariant models
class RRPlus_M3(torch.nn.Module):
    def __init__(self, use_bias=False):
        super(RRPlus_M3, self).__init__()
        # Parameters of the model
        use_bias = False
        eps = 2e-5
        n_channels = 256
        n_classes = 10

        # # G-conv approach
        group = eerie.Group('R1R+')

        # For first layer:
        N_h_RdG = 9
        base = 2
        h_grid_RdG = group.h_grid_global(N_h_RdG, base ** (N_h_RdG - 1))
        print(h_grid_RdG.grid)

        # Multi-scale interactions (@Erik)
        # For subsequent layers:
        N_h = 3  # <--- TODO: not sure if this is the most optimal though, but it reduces the h_axis nicely to size 1 in the last layer
        base = 2
        h_grid = group.h_grid_global(N_h, base ** (N_h - 1))
        print(h_grid.grid)
        n_channels_G = int(n_channels / 1.7) #int(n_channels / 1.75 * (N_h / N_h))  # For Nh=6 use 2.45, For Nh = 7, use 2.65, For Nh=8, use 2.85, For Nh=9, use 3.0,



        self.c1 = eerie.nn.GConvRdG(group, in_channels=1, out_channels=n_channels_G, kernel_size=79, h_grid=h_grid_RdG, bias=use_bias, stride=1)
        self.c2 = eerie.nn.GConvGG(group, in_channels=n_channels_G, out_channels=n_channels_G, kernel_size=3, h_grid=h_grid, bias=use_bias, stride=1, h_crop=True)
        self.c3 = eerie.nn.GConvGG(group, in_channels=n_channels_G, out_channels=n_classes, kernel_size=1, h_grid=h_grid, bias=use_bias, stride=1, h_crop=True)
        # Conv Layers
        #self.c1 = torch.nn.Conv1d(in_channels=1,          out_channels=n_channels, kernel_size=80, stride=4, padding=(80 // 2), dilation=1, bias=use_bias)
        #self.c2 = torch.nn.Conv1d(in_channels=n_channels, out_channels=n_channels, kernel_size=3,  stride=1, padding=(3 // 2),  dilation=1, bias=use_bias)
        #self.c3 = torch.nn.Conv1d(in_channels=n_channels, out_channels=n_classes,  kernel_size=1,  stride=1, padding=0,         dilation=1, bias=use_bias)
        # BatchNorm Layers
        self.bn1 = torch.nn.BatchNorm2d(num_features=n_channels_G, eps=eps)
        self.bn2 = torch.nn.BatchNorm2d(num_features=n_channels_G, eps=eps)

        # Pooling
        self.pool = eerie.functional.max_pooling_R1

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                m.weight.data.normal_(0, torch.prod(torch.Tensor(list(m.weight.shape)[1:]))**(-1/2))
                if use_bias: m.bias.data.fill_(0.0)

    def forward(self, x):
        # Instead of strided conv, we use normal conv and then pooling.
        out = self.c1(x)
        out = self.pool(out, kernel_size=4, stride=4, padding=0)
        out = torch.relu(self.bn1(out))
        # -----
        out = self.pool(out, kernel_size=4, stride=4, padding=0)
        out = torch.relu(self.bn2(self.c2(out)))
        out = self.pool(out, kernel_size=4, stride=4, padding=0)
        # Global pooling
        out = torch.mean(out, dim=-1, keepdim=True) # pool over the time axis
        out = self.c3(out)
        # Then turn into features per time point (merging scale and the channel axes)
        out = torch.max(out, dim=-2).values  # pool over the scale axis
        out = out.view(out.size(0), 10)
        return out


class RRPlus_M5(torch.nn.Module):
    def __init__(self, use_bias=False, dropout=False):
        super(RRPlus_M5, self).__init__()
        # Parameters of the model
        use_bias = False
        eps = 2e-5
        n_channels = 128
        n_classes = 10
        dp_rate = 0.3

        # # G-conv approach
        group = eerie.Group('R1R+')

        # For first layer:
        N_h_RdG = 9
        base = 2
        h_grid_RdG = group.h_grid_global(N_h_RdG, base ** (N_h_RdG - 1))
        print(h_grid_RdG.grid)

        # For subsequent layers:
        # N_h = 1
        # base = 2
        # h_grid = group.h_grid_global(N_h, base ** (N_h - 1))
        # print(h_grid.grid)
        # n_channels_G = int(n_channels / (N_h / N_h))   #  For Nh=6 use 2.45, For Nh = 7, use 2.65, For Nh=8, use 2.85, For Nh=9, use 3.0,

        # Multi-scale interactions (@Erik)
        # For subsequent layers:
        N_h = 3  # <--- TODO: not sure if this is the most optimal though, but it reduces the h_axis nicely to size 1 in the last layer
        base = 2
        h_grid = group.h_grid_global(N_h, base ** (N_h - 1))
        print(h_grid.grid)
        n_channels_G = 74 #int(n_channels / 1.75 * (N_h / N_h))  # For Nh=6 use 2.45, For Nh = 7, use 2.65, For Nh=8, use 2.85, For Nh=9, use 3.0,

        # Conv Layers
        self.c1 = eerie.nn.GConvRdG(group, in_channels=1,              out_channels=n_channels_G,     kernel_size=79, h_grid=h_grid_RdG, bias=use_bias, stride=1)
        self.c2 = eerie.nn.GConvGG(group, in_channels=n_channels_G,     out_channels=n_channels_G,     kernel_size=3,  h_grid=h_grid, bias=use_bias, stride=1, h_crop=True)
        self.c3 = eerie.nn.GConvGG(group, in_channels=n_channels_G,     out_channels=n_channels_G * 2, kernel_size=3, h_grid=h_grid, bias=use_bias, stride=1, h_crop=True)
        self.c4 = eerie.nn.GConvGG(group, in_channels=n_channels_G * 2, out_channels=n_channels_G * 4, kernel_size=3, h_grid=h_grid, bias=use_bias, stride=1, h_crop=True)
        self.c5 = eerie.nn.GConvGG(group, in_channels=n_channels_G * 4, out_channels=n_classes,      kernel_size=1,  h_grid=h_grid, bias=use_bias, stride=1, h_crop=True)
        # BatchNorm Layers
        self.bn1 = torch.nn.BatchNorm2d(num_features=n_channels_G,     eps=eps)
        self.bn2 = torch.nn.BatchNorm2d(num_features=n_channels_G,     eps=eps)
        self.bn3 = torch.nn.BatchNorm2d(num_features=n_channels_G * 2, eps=eps)
        self.bn4 = torch.nn.BatchNorm2d(num_features=n_channels_G * 4, eps=eps)

        # Pooling
        self.pool = eerie.functional.max_pooling_R1

        # DropOut
        self.dp = torch.nn.Dropout(dp_rate)
        self.dropout = dropout

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                m.weight.data.normal_(0, torch.prod(torch.Tensor(list(m.weight.shape)[1:]))**(-1/2))
                if use_bias: m.bias.data.fill_(0.0)

    def forward(self, x):
        # Instead of strided conv, we use normal conv and then pooling.
        out = self.c1(x)
        out = self.pool(out, kernel_size=4, stride=4, padding=0)
        out = torch.relu(self.bn1(out))
        if self.dropout: out = self.dp(out)
        # -----
        out = self.pool(out, kernel_size=4, stride=4, padding=0)
        out = torch.relu(self.bn2(self.c2(out)))
        if self.dropout: out = self.dp(out)
        out = self.pool(out, kernel_size=4, stride=4, padding=0)
        out = torch.relu(self.bn3(self.c3(out)))
        if self.dropout: out = self.dp(out)
        out = self.pool(out, kernel_size=4, stride=4, padding=0)
        out = torch.relu(self.bn4(self.c4(out)))
        if self.dropout: out = self.dp(out)
        out = self.pool(out, kernel_size=4, stride=4, padding=0)
        # Global pooling
        out = torch.mean(out, dim=-1, keepdim=True)  # pool over the time axis
        out = self.c5(out)
        # Then turn into features per time point (merging scale and the channel axes)
        out = torch.max(out, dim=-2).values  # pool over the scale axis
        out = out.view(out.size(0), 10)
        return out


class RRPlus_M11(torch.nn.Module):
    def __init__(self, use_bias=False):
        super(RRPlus_M11, self).__init__()
        # Parameters of the model
        use_bias = False
        eps = 2e-5
        n_channels = 64
        n_classes = 10

        # # G-conv approach
        group = eerie.Group('R1R+')

        # For first layer:
        N_h_RdG = 9
        base = 2
        h_grid_RdG = group.h_grid_global(N_h_RdG, base ** (N_h_RdG - 1))
        print(h_grid_RdG.grid)

        # For subsequent layers:
        N_h = 1
        base = 2
        h_grid = group.h_grid_global(N_h, base ** (N_h - 1))
        print(h_grid.grid)
        n_channels_G = int(n_channels / 1.25)  #  For Nh=6 use 2.45, For Nh = 7, use 2.65, For Nh=8, use 2.85, For Nh=9, use 3.0,

        N_h_crop = 3  # <--- TODO: not sure if this is the most optimal though, but it reduces the h_axis nicely to size 1 in the last layer
        base = 2
        h_grid_crop = group.h_grid_global(N_h_crop, base ** (N_h_crop - 1))
        print(h_grid_crop.grid)
        #n_channels_G = int(n_channels / 1.75 * (N_h / N_h))  # For Nh=6 use 2.45, For Nh = 7, use 2.65, For Nh=8, use 2.85, For Nh=9, use 3.0,

        # Conv Layers
        self.c1 = eerie.nn.GConvRdG(group, in_channels=1,              out_channels=n_channels_G,     kernel_size=79, h_grid=h_grid_RdG, bias=use_bias, stride=1)
        self.c2 = eerie.nn.GConvGG(group, in_channels=n_channels_G,     out_channels=n_channels_G,     kernel_size=3,  h_grid=h_grid_crop, bias=use_bias, stride=1, h_crop=True)
        self.c3 = eerie.nn.GConvGG(group, in_channels=n_channels_G,     out_channels=n_channels_G,     kernel_size=3,  h_grid=h_grid, bias=use_bias, stride=1)
        self.c4 = eerie.nn.GConvGG(group, in_channels=n_channels_G,     out_channels=n_channels_G * 2, kernel_size=3, h_grid=h_grid_crop, bias=use_bias, stride=1, h_crop=True)
        self.c5 = eerie.nn.GConvGG(group, in_channels=n_channels_G * 2, out_channels=n_channels_G * 2, kernel_size=3, h_grid=h_grid, bias=use_bias, stride=1)
        self.c6 = eerie.nn.GConvGG(group, in_channels=n_channels_G * 2, out_channels=n_channels_G * 4, kernel_size=3, h_grid=h_grid_crop, bias=use_bias, stride=1, h_crop=True)
        self.c7 = eerie.nn.GConvGG(group, in_channels=n_channels_G * 4, out_channels=n_channels_G * 4, kernel_size=3, h_grid=h_grid, bias=use_bias, stride=1)
        self.c8 = eerie.nn.GConvGG(group, in_channels=n_channels_G * 4, out_channels=n_channels_G * 4, kernel_size=3, h_grid=h_grid, bias=use_bias, stride=1)
        self.c9 = eerie.nn.GConvGG(group, in_channels=n_channels_G * 4, out_channels=n_channels_G * 8, kernel_size=3, h_grid=h_grid_crop, bias=use_bias, stride=1, h_crop=True)
        self.c10 = eerie.nn.GConvGG(group, in_channels=n_channels_G * 8, out_channels=n_channels_G * 8, kernel_size=3, h_grid=h_grid, bias=use_bias, stride=1)
        self.c11 = eerie.nn.GConvGG(group, in_channels=n_channels_G * 8, out_channels=n_classes,      kernel_size=1,  h_grid=h_grid, bias=use_bias, stride=1)
        # BatchNorm Layers
        self.bn1 = torch.nn.BatchNorm2d(num_features=n_channels_G, eps=eps)
        self.bn2 = torch.nn.BatchNorm2d(num_features=n_channels_G, eps=eps)
        self.bn3 = torch.nn.BatchNorm2d(num_features=n_channels_G, eps=eps)
        self.bn4 = torch.nn.BatchNorm2d(num_features=n_channels_G * 2, eps=eps)
        self.bn5 = torch.nn.BatchNorm2d(num_features=n_channels_G * 2, eps=eps)
        self.bn6 = torch.nn.BatchNorm2d(num_features=n_channels_G * 4, eps=eps)
        self.bn7 = torch.nn.BatchNorm2d(num_features=n_channels_G * 4, eps=eps)
        self.bn8 = torch.nn.BatchNorm2d(num_features=n_channels_G * 4, eps=eps)
        self.bn9 = torch.nn.BatchNorm2d(num_features=n_channels_G * 8, eps=eps)
        self.bn10 = torch.nn.BatchNorm2d(num_features=n_channels_G * 8, eps=eps)

        # Pooling
        self.pool = eerie.functional.max_pooling_R1

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                m.weight.data.normal_(0, torch.prod(torch.Tensor(list(m.weight.shape)[1:]))**(-1/2))
                if use_bias: m.bias.data.fill_(0.0)

    def forward(self, x):
        # Instead of strided conv, we use normal conv and then pooling.
        out = self.c1(x)
        out = self.pool(out, kernel_size=4, stride=4, padding=0)
        out = torch.relu(self.bn1(out))
        # -----
        out = self.pool(out, kernel_size=4, stride=4, padding=0)
        out = torch.relu(self.bn2(self.c2(out)))
        out = torch.relu(self.bn3(self.c3(out)))
        out = self.pool(out, kernel_size=4, stride=4, padding=0)
        out = torch.relu(self.bn4(self.c4(out)))
        out = torch.relu(self.bn5(self.c5(out)))
        out = self.pool(out, kernel_size=4, stride=4, padding=0)
        out = torch.relu(self.bn6(self.c6(out)))
        out = torch.relu(self.bn7(self.c7(out)))
        out = torch.relu(self.bn8(self.c8(out)))
        out = self.pool(out, kernel_size=4, stride=4, padding=0)
        out = torch.relu(self.bn9(self.c9(out)))
        out = torch.relu(self.bn10(self.c10(out)))
        # Global pooling
        out = torch.mean(out, dim=-1, keepdim=True) # pool over the time axis
        out = self.c11(out)
        # Then turn into features per time point (merging scale and the channel axes)
        out = torch.max(out, dim=-2).values  # pool over the scale axis
        out = out.view(out.size(0), 10)
        return out


class RRPlus_M18(torch.nn.Module):
    def __init__(self, use_bias=False):
        super(RRPlus_M18, self).__init__()
        # Parameters of the model
        use_bias = False
        eps = 2e-5
        n_channels = 64
        n_classes = 10

        # # G-conv approach
        group = eerie.Group('R1R+')

        # For first layer:
        N_h_RdG = 9
        base = 2
        h_grid_RdG = group.h_grid_global(N_h_RdG, base ** (N_h_RdG - 1))
        print(h_grid_RdG.grid)

        # For subsequent layers:
        N_h = 1
        base = 2
        h_grid = group.h_grid_global(N_h, base ** (N_h - 1))
        print(h_grid.grid)
        n_channels_G = int(n_channels / 1.12)  #  For Nh=6 use 2.45, For Nh = 7, use 2.65, For Nh=8, use 2.85, For Nh=9, use 3.0,

        N_h_crop = 3  # <--- TODO: not sure if this is the most optimal though, but it reduces the h_axis nicely to size 1 in the last layer
        base = 2
        h_grid_crop = group.h_grid_global(N_h_crop, base ** (N_h_crop - 1))
        print(h_grid_crop.grid)

        # Conv Layers
        self.c1 = eerie.nn.GConvRdG(group, in_channels=1,              out_channels=n_channels_G,     kernel_size=79, h_grid=h_grid_RdG, bias=use_bias, stride=1)
        self.c2 = eerie.nn.GConvGG(group, in_channels=n_channels_G,     out_channels=n_channels_G,     kernel_size=3, h_grid=h_grid_crop, bias=use_bias, stride=1, h_crop=True)
        self.c3 = eerie.nn.GConvGG(group, in_channels=n_channels_G,     out_channels=n_channels_G,     kernel_size=3, h_grid=h_grid, bias=use_bias, stride=1)
        self.c4 = eerie.nn.GConvGG(group, in_channels=n_channels_G,     out_channels=n_channels_G,     kernel_size=3, h_grid=h_grid, bias=use_bias, stride=1)
        self.c5 = eerie.nn.GConvGG(group, in_channels=n_channels_G,     out_channels=n_channels_G,     kernel_size=3, h_grid=h_grid, bias=use_bias, stride=1)
        self.c6 = eerie.nn.GConvGG(group, in_channels=n_channels_G,     out_channels=n_channels_G * 2, kernel_size=3, h_grid=h_grid_crop, bias=use_bias, stride=1, h_crop=True)
        self.c7 = eerie.nn.GConvGG(group, in_channels=n_channels_G * 2, out_channels=n_channels_G * 2, kernel_size=3, h_grid=h_grid, bias=use_bias, stride=1)
        self.c8 = eerie.nn.GConvGG(group, in_channels=n_channels_G * 2, out_channels=n_channels_G * 2, kernel_size=3, h_grid=h_grid, bias=use_bias, stride=1)
        self.c9 = eerie.nn.GConvGG(group, in_channels=n_channels_G * 2, out_channels=n_channels_G * 2, kernel_size=3, h_grid=h_grid, bias=use_bias, stride=1)
        self.c10 = eerie.nn.GConvGG(group, in_channels=n_channels_G * 2, out_channels=n_channels_G * 4, kernel_size=3, h_grid=h_grid_crop, bias=use_bias, stride=1, h_crop=True)
        self.c11 = eerie.nn.GConvGG(group, in_channels=n_channels_G * 4, out_channels=n_channels_G * 4, kernel_size=3, h_grid=h_grid, bias=use_bias, stride=1)
        self.c12 = eerie.nn.GConvGG(group, in_channels=n_channels_G * 4, out_channels=n_channels_G * 4, kernel_size=3, h_grid=h_grid, bias=use_bias, stride=1)
        self.c13 = eerie.nn.GConvGG(group, in_channels=n_channels_G * 4, out_channels=n_channels_G * 4, kernel_size=3, h_grid=h_grid, bias=use_bias, stride=1)
        self.c14 = eerie.nn.GConvGG(group, in_channels=n_channels_G * 4, out_channels=n_channels_G * 8, kernel_size=3, h_grid=h_grid_crop, bias=use_bias, stride=1, h_crop=True)
        self.c15 = eerie.nn.GConvGG(group, in_channels=n_channels_G * 8, out_channels=n_channels_G * 8, kernel_size=3, h_grid=h_grid, bias=use_bias, stride=1)
        self.c16 = eerie.nn.GConvGG(group, in_channels=n_channels_G * 8, out_channels=n_channels_G * 8, kernel_size=3, h_grid=h_grid, bias=use_bias, stride=1)
        self.c17 = eerie.nn.GConvGG(group, in_channels=n_channels_G * 8, out_channels=n_channels_G * 8, kernel_size=3, h_grid=h_grid, bias=use_bias, stride=1)
        self.c18 = eerie.nn.GConvGG(group, in_channels=n_channels_G * 8, out_channels=n_classes,      kernel_size=1,  h_grid=h_grid, bias=use_bias, stride=1)
        # BatchNorm Layers
        self.bn1 = torch.nn.BatchNorm2d(num_features=n_channels_G, eps=eps)
        self.bn2 = torch.nn.BatchNorm2d(num_features=n_channels_G, eps=eps)
        self.bn3 = torch.nn.BatchNorm2d(num_features=n_channels_G, eps=eps)
        self.bn4 = torch.nn.BatchNorm2d(num_features=n_channels_G, eps=eps)
        self.bn5 = torch.nn.BatchNorm2d(num_features=n_channels_G, eps=eps)
        self.bn6 = torch.nn.BatchNorm2d(num_features=n_channels_G * 2, eps=eps)
        self.bn7 = torch.nn.BatchNorm2d(num_features=n_channels_G * 2, eps=eps)
        self.bn8 = torch.nn.BatchNorm2d(num_features=n_channels_G * 2, eps=eps)
        self.bn9 = torch.nn.BatchNorm2d(num_features=n_channels_G * 2, eps=eps)
        self.bn10 = torch.nn.BatchNorm2d(num_features=n_channels_G * 4, eps=eps)
        self.bn11 = torch.nn.BatchNorm2d(num_features=n_channels_G * 4, eps=eps)
        self.bn12 = torch.nn.BatchNorm2d(num_features=n_channels_G * 4, eps=eps)
        self.bn13 = torch.nn.BatchNorm2d(num_features=n_channels_G * 4, eps=eps)
        self.bn14 = torch.nn.BatchNorm2d(num_features=n_channels_G * 8, eps=eps)
        self.bn15 = torch.nn.BatchNorm2d(num_features=n_channels_G * 8, eps=eps)
        self.bn16 = torch.nn.BatchNorm2d(num_features=n_channels_G * 8, eps=eps)
        self.bn17 = torch.nn.BatchNorm2d(num_features=n_channels_G * 8, eps=eps)

        # Pooling
        self.pool = eerie.functional.max_pooling_R1

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                m.weight.data.normal_(0, torch.prod(torch.Tensor(list(m.weight.shape)[1:]))**(-1/2))
                if use_bias: m.bias.data.fill_(0.0)

    def forward(self, x):
        # Instead of strided conv, we use normal conv and then pooling.
        out = self.c1(x)
        out = self.pool(out, kernel_size=4, stride=4, padding=0)
        out = torch.relu(self.bn1(out))
        # -----
        out = self.pool(out, kernel_size=4, stride=4, padding=0)
        out = torch.relu(self.bn2(self.c2(out)))
        out = torch.relu(self.bn3(self.c3(out)))
        out = torch.relu(self.bn4(self.c4(out)))
        out = torch.relu(self.bn5(self.c5(out)))
        out = self.pool(out, kernel_size=4, stride=4, padding=0)
        out = torch.relu(self.bn6(self.c6(out)))
        out = torch.relu(self.bn7(self.c7(out)))
        out = torch.relu(self.bn8(self.c8(out)))
        out = torch.relu(self.bn9(self.c9(out)))
        out = self.pool(out, kernel_size=4, stride=4, padding=0)
        out = torch.relu(self.bn10(self.c10(out)))
        out = torch.relu(self.bn11(self.c11(out)))
        out = torch.relu(self.bn12(self.c12(out)))
        out = torch.relu(self.bn13(self.c13(out)))
        out = self.pool(out, kernel_size=4, stride=4, padding=0)
        out = torch.relu(self.bn14(self.c14(out)))
        out = torch.relu(self.bn15(self.c15(out)))
        out = torch.relu(self.bn16(self.c16(out)))
        out = torch.relu(self.bn17(self.c17(out)))
        # Global pooling
        out = torch.mean(out, dim=-1, keepdim=True) # pool over the time axis
        out = self.c18(out)
        # Then turn into features per time point (merging scale and the channel axes)
        out = torch.max(out, dim=-2).values  # pool over the scale axis
        out = out.view(out.size(0), 10)
        return out


class RRPlus_IdentityBlock(torch.nn.Module):
    def __init__(self, group, in_channels, out_channels, kernel_size, h_grid, h_grid_crop, stride, use_bias, eps, override=False):
        super(RRPlus_IdentityBlock, self).__init__()

        # Additional params
        self.diff_size = (in_channels != out_channels)
        self.h_grid = h_grid_crop if self.diff_size else h_grid
        self.crop = h_grid_crop.N
        self.override = override

        # Conv Layers
        if not override:
            self.c1 = eerie.nn.GConvGG(group, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, h_grid=self.h_grid, stride=stride,
                                      bias=use_bias, h_crop=self.diff_size)
        else:
            self.c1 = eerie.nn.GConvGG(group, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, h_grid=h_grid_crop, stride=stride,
                                       bias=use_bias, h_crop=True)

        self.c2 = eerie.nn.GConvGG(group, in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, h_grid=h_grid, stride=stride,
                                  bias=use_bias, h_crop=False)

        # BatchNorm Layers
        self.bn1 = torch.nn.BatchNorm2d(num_features=out_channels, eps=eps)
        self.bn2 = torch.nn.BatchNorm2d(num_features=out_channels, eps=eps)
        self.bn_out = torch.nn.BatchNorm2d(num_features=out_channels, eps=eps)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                m.weight.data.normal_(0, torch.prod(torch.Tensor(list(m.weight.shape)[1:])) ** (-1 / 2))
                if use_bias: m.bias.data.fill_(0.0)

    def forward(self, x):
        out = torch.relu(self.bn1(self.c1(x)))
        out = self.bn2(self.c2(out))
        # shortcut
        if self.diff_size:
            out = out + x[:,:,:-(self.crop - 1),:].repeat(1, 2, 1, 1)
        elif self.override:
            out = out + x[:, :, :-(self.crop - 1), :]
        else:
            out = out + x
        out = torch.relu(self.bn_out(out))
        return out


class RRPlus_M34res(torch.nn.Module):
    def __init__(self, use_bias=False):
        super(RRPlus_M34res, self).__init__()
        # Parameters of the model
        use_bias = False
        eps = 2e-5
        n_channels = 48
        n_classes = 10

        # # G-conv approach
        group = eerie.Group('R1R+')

        # For first layer:
        N_h_RdG = 9
        base = 2
        h_grid_RdG = group.h_grid_global(N_h_RdG, base ** (N_h_RdG - 1))
        print(h_grid_RdG.grid)

        # For subsequent layers:
        N_h = 1
        base = 2
        h_grid = group.h_grid_global(N_h, base ** (N_h - 1))
        print(h_grid.grid)
        n_channels_G = int(45)  # For Nh=6 use 2.45, For Nh = 7, use 2.65, For Nh=8, use 2.85, For Nh=9, use 3.0,

        N_h_crop = 3  # <--- TODO: not sure if this is the most optimal though, but it reduces the h_axis nicely to size 1 in the last layer
        base = 2
        h_grid_crop = group.h_grid_global(N_h_crop, base ** (N_h_crop - 1))
        print(h_grid_crop.grid)

        # Conv Layers
        self.c1 = eerie.nn.GConvRdG(group, in_channels=1, out_channels=n_channels_G, kernel_size=79, h_grid=h_grid_RdG, bias=use_bias, stride=1)
        # ----
        first_block = []
        for i in range(3):
            override = True if i == 0 else False
            first_block.append(RRPlus_IdentityBlock(group, in_channels=n_channels_G, out_channels=n_channels_G, kernel_size=3, stride=1,
                                                    h_grid=h_grid, h_grid_crop=h_grid_crop, use_bias=use_bias, eps=eps, override=override))
        self.first_block = nn.Sequential(*first_block)
        # ----
        sec_block = []
        for i in range(4):
            b_channels = n_channels_G if i == 0 else n_channels_G * 2
            sec_block.append(RRPlus_IdentityBlock(group, in_channels=b_channels, out_channels=n_channels_G * 2, kernel_size=3, stride=1,
                                                  h_grid=h_grid, h_grid_crop=h_grid_crop, use_bias=use_bias, eps=eps))
        self.sec_block = nn.Sequential(*sec_block)
        # ----
        thrd_block = []
        for i in range(6):
            b_channels = n_channels_G * 2 if i == 0 else n_channels_G * 4
            thrd_block.append(RRPlus_IdentityBlock(group, in_channels=b_channels, out_channels=n_channels_G * 4, kernel_size=3, stride=1,
                                                   h_grid=h_grid, h_grid_crop=h_grid_crop, use_bias=use_bias, eps=eps))
        self.thrd_block = nn.Sequential(*thrd_block)
        # ----
        frth_block = []
        for i in range(3):
            b_channels = n_channels_G * 4 if i == 0 else n_channels_G * 8
            frth_block.append(RRPlus_IdentityBlock(group, in_channels=b_channels, out_channels=n_channels_G * 8, kernel_size=3, stride=1,
                                                   h_grid=h_grid, h_grid_crop=h_grid_crop, use_bias=use_bias, eps=eps))
        self.frth_block = nn.Sequential(*frth_block)
        # ----
        self.c_out = eerie.nn.GConvGG(group, in_channels=n_channels_G * 8, out_channels=n_classes, kernel_size=1, stride=1, h_grid=h_grid, bias=use_bias)

        # BatchNorm Layers
        self.bn1 = torch.nn.BatchNorm2d(num_features=n_channels_G, eps=eps)

        # Pooling
        self.pool = eerie.functional.max_pooling_R1

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                m.weight.data.normal_(0, torch.prod(torch.Tensor(list(m.weight.shape)[1:]))**(-1/2))
                if use_bias: m.bias.data.fill_(0.0)

    def forward(self, x):
        # Instead of strided conv, we use normal conv and then pooling.
        out = self.c1(x)
        out = self.pool(out, kernel_size=4, stride=4, padding=0)
        out = torch.relu(self.bn1(out))
        # -----
        out = self.pool(out, kernel_size=4, stride=4, padding=0)
        out = self.first_block(out)
        out = self.pool(out, kernel_size=4, stride=4, padding=0)
        out = self.sec_block(out)
        out = self.pool(out, kernel_size=4, stride=4, padding=0)
        out = self.thrd_block(out)
        out = self.pool(out, kernel_size=4, stride=4, padding=0)
        out = self.frth_block(out)
        # Global pooling
        out = torch.mean(out, dim=-1, keepdim=True) # pool over the time axis
        out = self.c_out(out)
        # Then turn into features per time point (merging scale and the channel axes)
        out = torch.max(out, dim=-2).values  # pool over the scale axis
        out = out.view(out.size(0), 10)
        return out


if __name__ == '__main__':
    from experiments.utils import num_params

    # Sanity check
    print('R_M3')
    model = R_M3()
    num_params(model)

    # Sanity check
    print('R_M5')
    model = R_M5()
    num_params(model)

    # Sanity check
    print('R_M11')
    model = R_M11()
    num_params(model)

    # Sanity check
    print('R_M18')
    model = R_M18()
    num_params(model)

    # Sanity check
    print('R_M34res')
    model = R_M34res()
    num_params(model)
    model(torch.rand([2, 1, 80000]))

    # Sanity check
    print('RR+_M3')
    model = RRPlus_M3()
    num_params(model)
    model(torch.rand([2, 1, 80000]))

    # Sanity check
    print('RR+_M5')
    model = RRPlus_M5()
    num_params(model)
    model(torch.rand([2, 1, 80000]))  # Sanity check

    # Sanity check
    print('RR+_M11')
    model = RRPlus_M11()
    num_params(model)

    # Sanity check
    print('RR+_M18')
    model = RRPlus_M18()
    num_params(model)
    model(torch.rand([2, 1, 80000]))  # Sanity check

    # Sanity check
    print('RR+_M34res')
    model = RRPlus_M34res()
    num_params(model)
    model(torch.rand([2, 1, 80000]))  # Sanity check