#-*- coding: utf-8 -*-

import abc
import glob
import os
import json
from scipy import io
import numpy as np
import h5py
import random
import re

random.seed(111)

class FileSorter:
    def __init__(self):
        pass
    
    def sort(self, list_of_strs):
        list_of_strs.sort(key=self._alphanum_key)

    def _tryint(self, s):
        try:
            return int(s)
        except:
            return s
    
    def _alphanum_key(self, s):
        """ Turn a string into a list of string and number chunks.
            "z23a" -> ["z", 23, "a"]
        """
        return [ self._tryint(c) for c in re.split('([0-9]+)', s) ]


class File(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self):
        pass
    
    @abc.abstractmethod
    def read(self, filename):
        pass
    
    @abc.abstractmethod
    def write(self, data, filename, write_mode="w"):
        pass
    
    def _check_directory(self, filename):
        directory = os.path.split(filename)[0]
        if directory != "" and not os.path.exists(directory):
            os.mkdir(directory)

class FileJson(File):
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
    
    # Todo : Exception ó��
    def write(self, data, filename, write_mode="w"):
        self._check_directory(filename)        
        with open(filename, write_mode) as f:
            json.dump(data, f, indent=4)


class FileMat(File):
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
    
    def write(self, data, filename, write_mode="w"):
        self._check_directory(filename)        
        io.savemat(filename, data)


# Todo : staticmethod??
class FileHDF5(File):
    def read(self, filename, db_name):
        db = h5py.File(filename, "r")
        np_data = np.array(db[db_name])
        db.close()
        
        return np_data
    
    def write(self, data, filename, db_name, write_mode="a", dtype="float"):
        """Write data to hdf5 format.
        
        Parameters
        ----------
        data : array
            data to write
            
        filename : str
            filename including path
            
        db_name : str
            database name
            
        """
        self._check_directory(filename)        
        # todo : overwrite check
        db = h5py.File(filename, write_mode)
        dataset = db.create_dataset(db_name, data.shape, dtype=dtype)
        dataset[:] = data[:]
        db.close()


def list_files(directory, pattern="*.*", n_files_to_sample=None, recursive_option=True, random_order=True):
    """list files in a directory matched in defined pattern.

    Parameters
    ----------
    directory : str
        filename of json file

    pattern : str
        regular expression for file matching
    
    n_files_to_sample : int or None
        number of files to sample randomly and return.
        If this parameter is None, function returns every files.
    
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
    
    FileSorter().sort(files)
        
    if n_files_to_sample is not None:
        if random_order:
            files = random.sample(files, n_files_to_sample)
        else:
            files = files[:n_files_to_sample]
    return files


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    
    


