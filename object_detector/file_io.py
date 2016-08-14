#-*- coding: utf-8 -*-

import glob
import os
import commentjson as json

def read_json(filename):
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

    conf = json.loads(open(filename).read())
    return conf 

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
    import doctest
    doctest.testmod()
    
    
    


