import argparse


def _get_parser():
    # Running settings
    parser = argparse.ArgumentParser(description='FMA experiments.')
    # Parse
    parser.add_argument('--model', type=str, default='RR+_M3', metavar='M', help='type of model to use {, , }')
    parser.add_argument('--batch_size', type=int, default=8, metavar='N',  help='input batch size for training (default: 8)')
    parser.add_argument("--device", type=str, default="cuda", help="Where to deploy the model {cuda, cpu}")
    parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 0)')
    parser.add_argument('--pretrained', default=False, action='store_true', help='use pre-trained model. If false, the model will be trained.')
    parser.add_argument('--wavelet_loss', default=False, action='store_true', help='use wavelet loss. If false, it wont be used.')
    parser.add_argument('--warm_up', default=False, action='store_true', help='warm-up model. If True, small lr will be used for few epochs.')
    parser.add_argument('--extra_comment', type=str, default="")
    # Return parser
    return parser


def parse_args():
    parser = _get_parser()
    return parser.parse_args()