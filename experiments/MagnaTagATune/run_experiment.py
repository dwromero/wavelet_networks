# surfsara
import sys
#sys.path.extend(['/scratch/'])  # uncomment line if in surfsara. add /scratch/ to dataset path as well.
# torch
import torch
import torch.nn as nn
# built-in
import copy
import numpy as np
import random
import os
# models
import experiments.MagnaTagATune.parser as parser
import experiments.MagnaTagATune.model as models
import experiments.MagnaTagATune.dataset as dataset
import experiments.MagnaTagATune.trainer as trainer
import experiments.MagnaTagATune.tester as tester
# logger
from experiments.logger import Logger

def main(args):
    # Parse arguments
    args = copy.deepcopy(args)

    # Fix seeds for reproducibility and comparability
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # Select model and parameters. Both are inline with the baseline parameters.
    if 'SampleCNN' in args.model:
        args.epochs = 100
        args.weight_decay = 1e-6
        args.lr = 1e-2
        args.optim = 'sgd'

        if args.model == 'SampleCNN':
            model = models.SampleCNN()
        elif args.model == 'RR+_SampleCNN':
            model = models.RRPlus_SampleCNN()

    # Define the device to be used and move model to that device ( :0 required for multiGPU)
    args.device = 'cuda:0' if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu'
    # Check if multi-GPU available and if so, use the available GPU's
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)  # Required for multi-GPU
    model.to(args.device)

    # Define transforms and create dataloaders
    dataloaders, test_loader = dataset.get_dataset(batch_size=args.batch_size, num_workers=4)

    # Create model directory and instantiate args.path
    model_directory(args)

    # Train the model
    if not args.pretrained:
        # Create logger
        # sys.stdout = Logger(args)
        # Print arguments (Sanity check)
        print(args)
        # Train the model
        import datetime
        print(datetime.datetime.now())
        trainer.train(model, dataloaders, args)

    # Test the model
    if args.pretrained: model.load_state_dict(torch.load(args.path))
    tester.test(model, test_loader, args.device)


def model_directory(args):
    # Create name from arguments
    comment = "model_{}_optim_{}_lr_{}_wd_{}_seed_{}/".format(args.model, args.optim, args.lr, args.weight_decay, args.seed)
    if args.extra_comment is not "": comment = comment[:-1] + "_" + args.extra_comment + comment[-1]
    # Create directory
    modeldir = "./saved/" + comment
    os.makedirs(modeldir, exist_ok=True)
    # Add the path to the args
    args.path = modeldir + "model.pth"


if __name__ == '__main__':
    main(parser.parse_args())
