#!/usr/bin/env python3
'''
AITrain backend
'''
# -*- encoding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import data_parser

import tensorflow as tf

def main(argv):
    data_parser.main()

if __name__ == '__main__':
    tf.app.run()
