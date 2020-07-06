import torch
import eerie

class MSLELoss(torch.nn.Module):
    def __init__(self, n_classes=10):
        super(MSLELoss, self).__init__()
        self.n_classes = n_classes

    def forward(self, predicted, actual):
        predicted = torch.softmax(predicted, dim=-1)
        actual_onehot = actual.reshape(-1, 1)
        actual_onehot = (actual_onehot == torch.arange(self.n_classes).cuda().reshape(1, self.n_classes)).float()

        return torch.mean((torch.log(predicted.float() + 1.) - torch.log(actual_onehot.float() + 1.)) ** 2)

class WaveletLoss(torch.nn.Module):
    def __init__(self, weight_loss=10):
        super(WaveletLoss, self).__init__()
        self.weight_loss = weight_loss

    def forward(self, model):
        loss = 0.0
        num_lyrs = 0

        # Go through modules that are instances of GConvs
        for m in model.modules():
            if not isinstance(m, eerie.nn.GConvRdG) and not(isinstance(m, eerie.nn.GConvGG)):
                continue
            if m.weights.shape[-1] == 1:
                continue
            if isinstance(m, eerie.nn.GConvRdG):
                index = -1
            elif isinstance(m, eerie.nn.GConvGG):
                index = (-2, -1)
            loss = loss + torch.mean(torch.sum(m.weights, dim=index)**2)
            num_lyrs += 1

        # Avoid division by 0
        if num_lyrs == 0:
            num_lyrs = 1

        loss = self.weight_loss * loss #/ float(num_lyrs)
        return loss



