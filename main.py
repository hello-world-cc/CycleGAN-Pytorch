import os
from argparse import ArgumentParser
import train
import torch
import test
import time


def get_args():
    parser = ArgumentParser(description='cycleGANv1 PyTorch')
    parser.add_argument('--crop_size', type=int, default=286, help='crop size for the CelebA dataset')
    parser.add_argument('--img_size', type=int, default=256, help='image resolution')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_idt', type=float, default=0.5, help='weight for gradient penalty')

    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=4, help='mini-batch size')
    parser.add_argument('--train_epoch', type=int, default=200, help='number of total iterations for training D')
    parser.add_argument('--g_lr', type=float, default=0.0002, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0002, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=1, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')

    # Test configuration.
    parser.add_argument('--test_epoch', type=int, default=200000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])

    # Directories.
    parser.add_argument('--img_path', type=str, default='../datasets/horse2zebra/')
    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--model_save_dir', type=str, default='./checkpoint')
    parser.add_argument('--model_save_epoch', type=int, default=10)


    args = parser.parse_args()
    return args


def main():
  args = get_args()

  if args.mode == "train":
      print("Training")
      start_time = time.time()
      model = train.cycleGAN(args)
      model.train(args)
      print("Total time:",(time.time()-start_time)/60.0/60.0,"hour")
  else:
      print("Testing")
      model = test.cycleGAN(args)
      model.test(args)


if __name__ == '__main__':
    main()














