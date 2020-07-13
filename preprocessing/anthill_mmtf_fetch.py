#!/usr/bin/env python3

from mmtf_util import *
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fetch a .mmtf file')
    parser.add_argument('url', help='url to fetch')
    parser.add_argument('target_location', help='download location')
    args = parser.parse_args()
    if os.path.isfile(args.target_location):
        filename = os.path.basename(args.target_location)
        print('cached file for {} exists, not redownloading'.format(filename))
    else:
        os.chdir(os.path.dirname(args.target_location))
        os.system('wget {}'.format(args.url))
