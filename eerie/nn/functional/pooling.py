import torch


def max_pooling_R1(input, kernel_size, stride, padding = 1):
    input_size = input.size()
    out = input.view(input_size[0], input_size[1] * input_size[2], input_size[3])
    out = torch.max_pool1d(out, kernel_size=kernel_size, stride=stride, padding=padding)
    out = out.view(input_size[0], input_size[1], input_size[2], out.size()[2])
    return out

def average_pooling_R1(input, kernel_size, stride, padding = 1):
    input_size = input.size()
    out = input.view(input_size[0], input_size[1] * input_size[2], input_size[3])
    out = torch.nn.functional.avg_pool1d(out, kernel_size=kernel_size, stride=stride, padding=padding)
    out = out.view(input_size[0], input_size[1], input_size[2], out.size()[2], out.size()[3])
    return out