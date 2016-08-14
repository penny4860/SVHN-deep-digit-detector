#-*- coding: utf-8 -*-

import commentjson as json

def load_conf(filename):
    """This class provides image scanning interfaces of sliding window concept.

    Parameters
    ----------
    filename : str
        filename of json file

    Returns
    ----------
    conf : dict
        dictionary containing contents of json file

    Examples
    --------
    >>> import object_detector.conf as conf
    >>> conf_ = conf.load_conf("..//conf//cars.json")
    >>> type(conf_)
    <type 'dict'>
    >>> conf_['image_dataset'][conf_['image_dataset'].rfind("/") + 1:]
    u'car_side'
    """

    conf = json.loads(open(filename).read())
    return conf 

if __name__ == "__main__":
    import doctest
    doctest.testmod()




