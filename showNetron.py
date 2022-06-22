#!/usr/bin/env python3

import argparse
import netron

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str)
    args = parser.parse_args()
    print(args)
    netron.start(args.filename)
