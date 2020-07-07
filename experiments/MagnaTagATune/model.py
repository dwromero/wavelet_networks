import torch
import torch.nn as nn
import experiments.MagnaTagATune.config as config
import eerie


class SampleCNN(nn.Module):
    def __init__(self):
        super(SampleCNN, self).__init__()

        # 59049 x 1
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU())
        # 19683 x 128
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3))
        # 6561 x 128
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(3,stride=3))
        # 2187 x 128
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3,stride=3))
        # 729 x 256
        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3,stride=3))
        # 243 x 256
        self.conv6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3,stride=3),
            nn.Dropout(config.DROPOUT))
        # 81 x 256
        self.conv7 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3,stride=3))
        # 27 x 256
        self.conv8 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3,stride=3))
        # 9 x 256
        self.conv9 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3,stride=3))
        # 3 x 256
        self.conv10 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(3,stride=3))
        # 1 x 512 
        self.conv11 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT))
        # 1 x 512 
        self.fc = nn.Linear(512, 50)
        self.activation = nn.Sigmoid()
    
    def forward(self, x):
        # input x : 23 x 59049 x 1
        # expected conv1d input : minibatch_size x num_channel x width

        x = x.view(x.shape[0], 1,-1)
        # x : 23 x 1 x 59049

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.conv8(out)
        out = self.conv9(out)
        out = self.conv10(out)
        out = self.conv11(out) 
        
        out = out.view(x.shape[0], out.size(1) * out.size(2))
        logit = self.fc(out)

        #logit = self.activation(logit)

        return logit


class RRPlus_SampleCNN(nn.Module):
    def __init__(self):
        super(RRPlus_SampleCNN, self).__init__()
        # Parameters of the model
        n_channels = 128
        use_bias = False
        kernel_size = 3
        n_tags = 50

        # # G-conv approach
        group = eerie.Group('R1R+')

        # For first layer:
        N_h_RdG = 9
        base = 2
        h_grid_RdG = group.h_grid_global(N_h_RdG, base ** (N_h_RdG - 1))
        print(h_grid_RdG.grid)
        n_channels_G = int(90)

        # For subsequent layers:
        N_h = 1
        base = 2
        h_grid = group.h_grid_global(N_h, base ** (N_h - 1))
        print(h_grid.grid)

        N_h_crop = 3  # <--- TODO: not sure if this is the most optimal though, but it reduces the h_axis nicely to size 1 in the last layer
        base = 2
        h_grid_crop = group.h_grid_global(N_h_crop, base ** (N_h_crop - 1))
        print(h_grid_crop.grid)

        # Pooling
        self.pool = eerie.functional.max_pooling_R1
        # DropOut
        self.dropout = torch.nn.Dropout(config.DROPOUT)

        # 59049 x 1
        self.conv1 = eerie.nn.GConvRdG(group, in_channels=1, out_channels=n_channels_G, kernel_size=kernel_size, h_grid=h_grid_RdG, bias=use_bias, stride=1)
        self.bn1 = torch.nn.BatchNorm2d(num_features=n_channels_G)

        # 19683 x 128
        self.conv2 = eerie.nn.GConvGG(group, in_channels=n_channels_G, out_channels=n_channels_G, kernel_size=kernel_size, h_grid=h_grid_crop, bias=use_bias, stride=1, h_crop=True)
        self.bn2 = torch.nn.BatchNorm2d(num_features=n_channels_G)

        # 6561 x 128
        self.conv3 = eerie.nn.GConvGG(group, in_channels=n_channels_G, out_channels=n_channels_G, kernel_size=kernel_size, h_grid=h_grid, bias=use_bias, stride=1)
        self.bn3 = torch.nn.BatchNorm2d(num_features=n_channels_G)

        # 2187 x 128
        self.conv4 = eerie.nn.GConvGG(group, in_channels=n_channels_G, out_channels=n_channels_G * 2, kernel_size=kernel_size, h_grid=h_grid, bias=use_bias, stride=1)
        self.bn4 = torch.nn.BatchNorm2d(num_features=n_channels_G * 2)

        # 729 x 256
        self.conv5 = eerie.nn.GConvGG(group, in_channels=n_channels_G * 2, out_channels=n_channels_G * 2, kernel_size=kernel_size, h_grid=h_grid_crop, bias=use_bias, stride=1, h_crop=True)
        self.bn5 = torch.nn.BatchNorm2d(num_features=n_channels_G * 2)

        # 243 x 256
        self.conv6 = eerie.nn.GConvGG(group, in_channels=n_channels_G * 2, out_channels=n_channels_G * 2, kernel_size=kernel_size, h_grid=h_grid, bias=use_bias, stride=1)
        self.bn6 = torch.nn.BatchNorm2d(num_features=n_channels_G * 2)

        # 81 x 256
        self.conv7 = eerie.nn.GConvGG(group, in_channels=n_channels_G * 2, out_channels=n_channels_G * 2, kernel_size=kernel_size, h_grid=h_grid, bias=use_bias, stride=1)
        self.bn7 = torch.nn.BatchNorm2d(num_features=n_channels_G * 2)

        # 27 x 256
        self.conv8 = eerie.nn.GConvGG(group, in_channels=n_channels_G * 2, out_channels=n_channels_G * 2, kernel_size=kernel_size, h_grid=h_grid_crop, bias=use_bias, stride=1, h_crop=True)
        self.bn8 = torch.nn.BatchNorm2d(num_features=n_channels_G * 2)

        # 9 x 256
        self.conv9 = eerie.nn.GConvGG(group, in_channels=n_channels_G * 2, out_channels=n_channels_G * 2, kernel_size=kernel_size, h_grid=h_grid, bias=use_bias, stride=1)
        self.bn9 = torch.nn.BatchNorm2d(num_features=n_channels_G * 2)

        # 3 x 256
        self.conv10 = eerie.nn.GConvGG(group, in_channels=n_channels_G * 2, out_channels=n_channels_G * 4, kernel_size=kernel_size, h_grid=h_grid, bias=use_bias, stride=1)
        self.bn10 = torch.nn.BatchNorm2d(num_features=n_channels_G * 4)

        # 1 x 512
        self.conv11 = eerie.nn.GConvGG(group, in_channels=n_channels_G * 4, out_channels=n_channels_G * 4, kernel_size=kernel_size, h_grid=h_grid_crop, bias=use_bias, stride=1, h_crop=True)
        self.bn11 = torch.nn.BatchNorm2d(num_features=n_channels_G * 4)

        # 1 x 512
        self.fc = nn.Linear(n_channels_G * 4, n_tags)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # input x : 23 x 59049 x 1
        # expected conv1d input : minibatch_size x num_channel x width

        x = x.view(x.shape[0], 1, -1)
        # x : 23 x 1 x 59049
        # -----
        out = self.conv1(x)
        out = self.pool(out, kernel_size=3, stride=3, padding=0)
        out = torch.relu(self.bn1(out))
        # -----
        out = torch.relu(self.bn2(self.conv2(out)))
        out = self.pool(out, kernel_size=3, stride=3, padding=0)

        out = torch.relu(self.bn3(self.conv3(out)))
        out = self.pool(out, kernel_size=3, stride=3, padding=0)

        out = torch.relu(self.bn4(self.conv4(out)))
        out = self.pool(out, kernel_size=3, stride=3, padding=0)

        out = torch.relu(self.bn5(self.conv5(out)))
        out = self.pool(out, kernel_size=3, stride=3, padding=0)

        out = torch.relu(self.bn6(self.conv6(out)))
        out = self.pool(out, kernel_size=3, stride=3, padding=0)
        out = self.dropout(out)

        out = torch.relu(self.bn7(self.conv7(out)))
        out = self.pool(out, kernel_size=3, stride=3, padding=0)

        out = torch.relu(self.bn8(self.conv8(out)))
        out = self.pool(out, kernel_size=3, stride=3, padding=0)

        out = torch.relu(self.bn9(self.conv9(out)))
        out = self.pool(out, kernel_size=3, stride=3, padding=0)

        out = torch.relu(self.bn10(self.conv10(out)))
        out = self.pool(out, kernel_size=3, stride=3, padding=0)

        out = torch.relu(self.bn11(self.conv11(out)))
        out = torch.max(out, dim=-2).values  # pool over the scale axis
        out = self.dropout(out)

        out = out.view(x.shape[0], out.size(1) * out.size(2))
        logit = self.fc(out)

        # logit = self.activation(logit)

        return logit

if __name__ == '__main__':
    from experiments.utils import num_params

    # Sanity check
    print('SampleCNN')
    model = SampleCNN()
    num_params(model)

    print('RR+_SampleCNN')
    model = RRPlus_SampleCNN()
    num_params(model)
