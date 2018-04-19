#!/usr/bin/env python3
'''
AITrain backend
'''
# -*- encoding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf

import data_parser

def main(argv):
    player_list = data_parser.import_data()

    print(argv)
    print(player_list[0])

if __name__ == '__main__':
    main = main
    argv = [evaluation=False]

    tf.app.run(main=main, argv=argv) 