#-*- coding: utf-8 -*-

import abc
import glob
import os
import commentjson as json
from scipy import io
import numpy as np
import h5py


class ReadFile(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self):
        pass
    
    @abc.abstractmethod
    def read(self, filename):
        pass

class ReadJson(ReadFile):
    def read(self, filename):
        """load json file as dict object

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
        """
        return json.loads(open(filename).read())

class ReadMat(ReadFile):
    def read(self, filename):
        """load mat file as dict object

        Parameters
        ----------
        filename : str
            filename of json file
    
        Returns
        ----------
        conf : dict
            dictionary containing contents of mat file
    
        Examples
        --------
        """
        return io.loadmat(filename)

# Todo : staticmethod??
class ReadHDF5(ReadFile):
    def read(self, filename, db_name):
        db = h5py.File(filename, "r")
        data = db[db_name]
        db.close()
     
        return data

# Todo : doctest have to be added
def list_files(directory, pattern="*.*", recursive_option=True):
    """list files in a directory matched in defined pattern.

    Parameters
    ----------
    directory : str
        filename of json file

    pattern : str
        regular expression for file matching
        
    recursive_option : boolean
        option for searching subdirectories. If this option is True, 
        function searches all subdirectories recursively.
        
    Returns
    ----------
    conf : dict
        dictionary containing contents of json file

    Examples
    --------
    """

    if recursive_option == True:
        dirs = [path for path, _, _ in os.walk(directory)]
    else:
        dirs = [directory]
    
    files = []
    for dir_ in dirs:
        for p in glob.glob(os.path.join(dir_, pattern)):
            files.append(p)
    return files

if __name__ == "__main__":
#     import doctest
#     doctest.testmod()
    
    # write test code
    data = np.arange(0, 100).reshape(10, 5, 2)
    db = h5py.File("test.hdf5", "w")
    dataset = db.create_dataset("test", (10, 5, 2), dtype="float")
    dataset[:] = data[:]
    print dataset
    db.close()

    # read test code    
    db = h5py.File("test.hdf5", "r")
    print db["test"][:,:]
    data = np.array(db["test"])
    print data
    db.close()
    
 
    


