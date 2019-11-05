import tensorflow as tf
import numpy as np
import math
from types import MethodType, FunctionType
import argparse


class Config:
    def __init__(self):
        self.batch_size = 200
        self.lr = 0.0002
        self.epoches = 200
        self.save_path = 'models/p29_gan_sin/gan_sin'
        self.eps = 1e-5

    def to_dict(self):
        result = {}
        for name in dir(self):
            if name.startswith('__'):
                continue
            attr = getattr(self, name)
            if isinstance(attr, MethodType) or isinstance(attr, FunctionType):
                continue
            print(name, type(attr))
            result[name] = attr
        return result

    def from_cmd_line(self):
        parser = argparse.ArgumentParser()
        attrs = self.to_dict()
        for name in attrs:
            attr = attrs[name]
            parser.add_argument('--' + name, type=type(attr), help='Default to %s' % attr, default=attr)

        a = parser.parse_args()

        for name in dir(a):
            if name in attrs:
                setattr(self, name, getattr(a, name))

    def __repr__(self):
        result = '{'
        attrs = self.to_dict()
        for name in attrs:
            result += '%s: %s; ' % (name, attrs[name])
        return result + '}'


class Tensors:
    pass


class Samples:
    pass
# def get_samples(batch_size):
#     start = -math.pi
#     end = math.pi
#
#     seg = (end - start) / batch_size
#
#     xs = []
#     ys = []
#     for _ in range(batch_size):
#        xs.append(start)
#        ys.append(math.sin(start))
#        start += seg
#
#     return xs, ys


class Sin:
    def __init__(self):
        pass

    def train(self):
        pass

    def predict(self, x):
        return 0.1234567

