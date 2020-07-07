# torch
import torch
import torch.nn as nn
# built-in
import functools
# project
import eerie


class OneDCNN(torch.nn.Module):
    def __init__(self, use_bias=False):
        super(OneDCNN, self).__init__()
        # Parameters of the model
        use_bias = False
        eps = 2e-5
        n_channels = 16
        n_classes = 10
        out_space_dim = 6
        dp_rate = 0.25

        # Conv Layers
        self.c1 = torch.nn.Conv1d(in_channels=1,              out_channels=n_channels,      kernel_size=64, stride=2, padding=(64 // 2), dilation=1, bias=use_bias)
        self.c2 = torch.nn.Conv1d(in_channels=n_channels,     out_channels=n_channels * 2,  kernel_size=32, stride=2, padding=(32 // 2), dilation=1, bias=use_bias)
        self.c3 = torch.nn.Conv1d(in_channels=n_channels * 2, out_channels=n_channels * 4,  kernel_size=16, stride=2, padding=(16 // 2), dilation=1, bias=use_bias)
        self.c4 = torch.nn.Conv1d(in_channels=n_channels * 4, out_channels=n_channels * 8,  kernel_size=8,  stride=2, padding=(8 // 2),  dilation=1, bias=use_bias)
        self.c5 = torch.nn.Conv1d(in_channels=n_channels * 8, out_channels=n_channels * 16, kernel_size=4,  stride=2, padding=(4 // 2),  dilation=1, bias=use_bias)

        # Fully connected
        self.f1 = torch.nn.Linear(in_features=n_channels * 16 * out_space_dim, out_features=n_channels * 8, bias=True)
        self.f2 = torch.nn.Linear(in_features=n_channels * 8,                  out_features=n_channels * 4, bias=True)
        self.f3 = torch.nn.Linear(in_features=n_channels * 4,                  out_features=n_classes,      bias=True)

        # BatchNorm Layers
        self.bn1 = torch.nn.BatchNorm1d(num_features=n_channels,      eps=eps)
        self.bn2 = torch.nn.BatchNorm1d(num_features=n_channels * 2,  eps=eps)
        self.bn3 = torch.nn.BatchNorm1d(num_features=n_channels * 4,  eps=eps)
        self.bn4 = torch.nn.BatchNorm1d(num_features=n_channels * 8,  eps=eps)
        self.bn5 = torch.nn.BatchNorm1d(num_features=n_channels * 16, eps=eps)

        # Pooling
        self.pool = torch.max_pool1d
        # DropOut
        self.dropout = torch.nn.Dropout(p=dp_rate)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                m.weight.data.normal_(0, torch.prod(torch.Tensor(list(m.weight.shape)[1:]))**(-1/2))
                if use_bias: m.bias.data.fill_(0.0)

    def forward(self, x):
        # Conv-layers
        out = self.bn1(torch.relu(self.c1(x)))
        out = self.pool(out, kernel_size=8, stride=8, padding=0)
        out = self.bn2(torch.relu(self.c2(out)))
        out = self.pool(out, kernel_size=8, stride=8, padding=0)
        out = self.bn3(torch.relu(self.c3(out)))
        out = self.bn4(torch.relu(self.c4(out)))
        out = self.bn5(torch.relu(self.c5(out)))
        out = self.pool(out, kernel_size=5, stride=5, padding=0)

        # Fully connected lyrs
        out = out.view(out.size(0), -1)
        out = self.dropout(self.f1(out))
        out = self.dropout(self.f2(out))
        out = self.f3(out)

        return out


class RRPlus_OneDCNN(torch.nn.Module):
    def __init__(self, use_bias=False):
        super(RRPlus_OneDCNN, self).__init__()
        # Parameters of the model
        use_bias = False
        eps = 2e-5
        n_channels = 12
        n_classes = 10
        out_space_dim = 6
        dp_rate = 0.25

        # # G-conv approach
        group = eerie.Group('R1R+')

        # For first layer:
        N_h_RdG = 9
        base = 2
        h_grid_RdG = group.h_grid_global(N_h_RdG, base ** (N_h_RdG - 1))
        print(h_grid_RdG.grid)

        N_h_crop = 3  # <--- TODO: not sure if this is the most optimal though, but it reduces the h_axis nicely to size 1 in the last layer
        base = 2
        h_grid_crop = group.h_grid_global(N_h_crop, base ** (N_h_crop - 1))
        print(h_grid_crop.grid)

        # Conv Layers
        self.c1 = eerie.nn.GConvRdG(group, in_channels=1,              out_channels=n_channels,     kernel_size=63, h_grid=h_grid_RdG, bias=use_bias, stride=1)
        self.c2 = eerie.nn.GConvGG(group, in_channels=n_channels,     out_channels=n_channels * 2,  kernel_size=31, h_grid=h_grid_crop, bias=use_bias, stride=1, h_crop=True)
        self.c3 = eerie.nn.GConvGG(group, in_channels=n_channels * 2, out_channels=n_channels * 4,  kernel_size=15, h_grid=h_grid_crop, bias=use_bias, stride=1, h_crop=True)
        self.c4 = eerie.nn.GConvGG(group, in_channels=n_channels * 4, out_channels=n_channels * 8,  kernel_size=7,  h_grid=h_grid_crop, bias=use_bias, stride=1, h_crop=True)
        self.c5 = eerie.nn.GConvGG(group, in_channels=n_channels * 8, out_channels=n_channels * 16, kernel_size=3,  h_grid=h_grid_crop, bias=use_bias, stride=1, h_crop=True)

        # Fully connected
        self.f1 = torch.nn.Linear(in_features=n_channels * 16 * out_space_dim, out_features= n_channels * 8, bias=True)
        self.f2 = torch.nn.Linear(in_features=n_channels * 8, out_features=n_channels * 4,  bias=True)
        self.f3 = torch.nn.Linear(in_features=n_channels * 4,  out_features=n_classes,      bias=True)

        # BatchNorm Layers
        self.bn1 = torch.nn.BatchNorm2d(num_features=n_channels,      eps=eps)
        self.bn2 = torch.nn.BatchNorm2d(num_features=n_channels * 2,  eps=eps)
        self.bn3 = torch.nn.BatchNorm2d(num_features=n_channels * 4,  eps=eps)
        self.bn4 = torch.nn.BatchNorm2d(num_features=n_channels * 8,  eps=eps)
        self.bn5 = torch.nn.BatchNorm2d(num_features=n_channels * 16, eps=eps)

        # Pooling
        self.pool = eerie.functional.max_pooling_R1
        # DropOut
        self.dropout = torch.nn.Dropout(p=dp_rate)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                m.weight.data.normal_(0, torch.prod(torch.Tensor(list(m.weight.shape)[1:]))**(-1/2))
                if use_bias: m.bias.data.fill_(0.0)

    def forward(self, x):
        # Conv-layers
        # We replace strided convolutions with normal convolutions followed by max pooling.
        # -----
        out = self.c1(x)
        out = self.pool(out, kernel_size=2, stride=2, padding=0)
        out = self.bn1(torch.relu(out))
        # -----
        out = self.pool(out, kernel_size=8, stride=8, padding=0)
        # -----
        out = self.c2(out)
        out = self.pool(out, kernel_size=2, stride=2, padding=0)
        out = self.bn2(torch.relu(out))
        # -----
        out = self.pool(out, kernel_size=8, stride=8, padding=0)
        # -----
        out = self.c3(out)
        out = self.pool(out, kernel_size=2, stride=2, padding=0)
        out = self.bn3(torch.relu(out))
        # -----
        out = self.c4(out)
        out = self.pool(out, kernel_size=2, stride=2, padding=0)
        out = self.bn4(torch.relu(out))
        # -----
        out = self.c5(out)
        out = self.pool(out, kernel_size=2, stride=2, padding=0)
        out = self.bn5(torch.relu(out))
        # -----
        out = self.pool(out, kernel_size=5, stride=5, padding=0)
        # -----
        # Fully connected lyrs
        out = out.view(out.size(0), -1)
        out = self.dropout(self.f1(out))
        out = self.dropout(self.f2(out))
        out = self.f3(out)

        return out


if __name__ == '__main__':
    from experiments.utils import num_params

    # Sanity check
    print('OneDCNN')
    model = OneDCNN()
    num_params(model)
    model(torch.rand([2, 1, 64000]))

    # Sanity check
    print('RR+_OneDCNN')
    model = RRPlus_OneDCNN()
    num_params(model)
    #model(torch.rand([2, 1, 50999]))
