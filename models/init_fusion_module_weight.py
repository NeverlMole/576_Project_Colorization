import torch
import model_helper
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '-f',
        '--full_image',
        required=True,
        help='The filename of full-image network to load')
    parser.add_argument(
        '-i',
        '--instance',
        required=True,
        help='The filename of instance network to load')
    parser.add_argument(
        '-o',
        '--output',
        required=True,
        help='The filename of output network to store')
    args = parser.parse_args()

    model = model_helper.get_fusion_model_for_training(args.full_image, args.instance)
    model_helper.save_model(model, args.output)