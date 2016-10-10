#-*- coding: utf-8 -*-

import os
import object_detector.file_io as file_io
import shutil
import tempfile
import numpy as np

def generate_file_and_data(data_type):
    test_root_dir = tempfile.mkdtemp()
    test_file = "test.json"
    if data_type == "dict":
        test_data = {"a":1, "b":2, "c":3}
    elif data_type == "array":
        test_data = np.random.rand(10,10)
        
    return os.path.join(test_root_dir, test_file), test_data


def test_list_files_at_non_recursive_option():

    # Setup    
    tmp_dir = "TestDir"
    os.mkdir(tmp_dir)
    files = [os.path.join(tmp_dir, afile) for afile in ["a.txt", "b.txt", "c.txt"]]
    for filename in files:
        with open(filename, "w") as _: pass

    # When
    files_listed = file_io.list_files(directory=tmp_dir, pattern="*.txt", n_files_to_sample=None, recursive_option=True)
    
    # Should
    assert set(files) == set(files_listed)
    
    # Clean up
    shutil.rmtree(tmp_dir)
    


def test_list_files():
    # Given the following directories and files
    test_root_dir = tempfile.mkdtemp()
    os.mkdir(test_root_dir + "\\sub1")
    os.mkdir(test_root_dir + "\\sub2")
     
    files_in_txt = [test_root_dir+ "\\a.txt", 
                    test_root_dir+ "\\b.txt", 
                    test_root_dir+ "\\sub1\\a_in_sub1.txt", 
                    test_root_dir+ "\\sub2\\b_in_sub2.txt"]
    files_in_log = ["test.log"]
     
    for filename in files_in_txt + files_in_log:
        with open(filename, "w") as _: pass
     
    # When perform list_files()
    files_listed_in_txt = file_io.list_files(directory=test_root_dir, pattern="*.txt", n_files_to_sample=None, recursive_option=True)
     
    # Then it should be have same elements with files_created.
    assert set(files_listed_in_txt) == set(files_in_txt)
     
    # Remove test files and directory
    shutil.rmtree(test_root_dir)


def test_FileJson_interface():
    # Given the following directory and dictionary data
    test_file, test_data = generate_file_and_data(data_type="dict")
     
    # When perform write it to file and read
    file_io.FileJson().write(test_data, test_file)
    read_data = file_io.FileJson().read(test_file)
     
    # Then it should be equal to the original test_data
    assert test_data == read_data
 
    # Remove test files and directory
    shutil.rmtree(os.path.dirname(test_file))

    
def test_FileMat_interface():
    # Given the following directory and dictionary data
    test_file, test_data = generate_file_and_data(data_type="dict")
     
    # When perform write it to file and read
    file_io.FileMat().write(test_data, test_file)
    read_data = file_io.FileMat().read(test_file)
    shutil.rmtree(os.path.dirname(test_file))
 
    read_data = {key_: value_ for key_, value_ in zip(read_data.keys(), read_data.values()) if key_[:2] != "__" and key_[-2:] != "__"}
     
    # Then it should be equal to the original test_data
    assert test_data.keys() == read_data.keys()
    assert test_data.values() == [val[0][0] for val in read_data.values()]

     
def test_FileHDF5_interface():
    # Given the following directory and dictionary data
    test_file, test_data = generate_file_and_data(data_type="array")
     
    # When perform write it to file and read
    file_io.FileHDF5().write(test_data, test_file, "test_db")
    read_data = file_io.FileHDF5().read(test_file, "test_db")
    shutil.rmtree(os.path.dirname(test_file))
     
    # Then it should be equal to the original test_data
    assert test_data.all() == read_data.all()


import pytest
if __name__ == '__main__':
    pytest.main([__file__])

    
    
