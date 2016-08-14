#-*- coding: utf-8 -*-

import abc

class Descriptor(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self):
        self._describe_behavior = None

    def describe(self):
        self._describe_behavior.operate()

class HOG(Descriptor):
    def __init__(self):
        self._describe_behavior = HOGDescriber()
    
class ColorHist(Descriptor):
    def __init__(self):
        self._describe_behavior = ColorHistDescriber()

class LBP(Descriptor):
    def __init__(self):
        self._describe_behavior = LBPDescriber()

class HuMoments(Descriptor):
    def __init__(self):
        self._describe_behavior = HuMomentsDescriber()


class Describer(object):
    def __init__(self):
        pass

    def operate(self):
        pass

class HOGDescriber(object):
    def __init__(self):
        pass

    def operate(self):
        pass

class ColorHistDescriber(object):
    def __init__(self):
        pass

    def operate(self):
        pass

class LBPDescriber(object):
    def __init__(self):
        pass

    def operate(self):
        pass

class HuMomentsDescriber(object):
    def __init__(self):
        pass

    def operate(self):
        pass
