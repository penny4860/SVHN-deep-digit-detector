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
        np_data = np.array(db[db_name])
        db.close()
        
        return np_data
    
    def write(self, data, filename, db_name, overwrite=True):
        """Write data to hdf5 format.
        
        Parameters
        ----------
        data : array
            data to write
            
        filename : str
            filename including path
            
        db_name : str
            database name
            
        overwrite : boolean
            overwrite option if 'filename' is already exists
        """
        if overwrite:
            write_mode = "a"
        else:
            write_mode = "w"
        
        db = h5py.File(filename, write_mode)
        dataset = db.create_dataset(db_name, data.shape, dtype="float")
        dataset[:] = data[:]
        db.close()

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
    
    data = np.arange(0, 100).reshape(10, 5, 2)
    
    ReadHDF5().write(data, "test2.hdf5", "test")
    print "done"
    data = ReadHDF5().read("test2.hdf5", "test")
    print data
    


