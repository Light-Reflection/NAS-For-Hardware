"""
make mask of face and complete them
"""
import os
import argparse
parse = argparse.ArgumentParser('complete')
parse.add_argument('--out_dir', type=str, help='the path of out dir')
parse.add_argument('--make_type', type=str)
args = parse.parse_args()






def complete():
    make_dir(os.path.join(args.out_dir, 'hats_imgs'))
    make_dir(os.path.join(args.out_dir, 'completed'))
    make_dir(os.path.join(args.out_dir, 'logs'))

    nImgs = len()





