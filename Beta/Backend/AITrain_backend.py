#!/usr/bin/env python3
'''
AITrain backend
'''
# -*- encoding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


import numpy as np
import pandas as pd

import tensorflow as tf

import data_parser

def main(argv):
    player_list = data_parser.import_data()

    print(player_list[0])

if __name__ == '__main__':
    tf.app.run()
