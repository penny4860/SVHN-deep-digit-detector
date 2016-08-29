#-*- coding: utf-8 -*-

import os
import object_detector.file_io as file_io
import shutil
import tempfile


def generate_test_file_and_data():
    test_root_dir = tempfile.mkdtemp()
    test_file = "test.json"
    test_data = {"a":1, "b":2, "c":3}
    return os.path.join(test_root_dir, test_file), test_data


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
    test_file, test_data = generate_test_file_and_data()
    
    # When perform write it to file and read
    file_io.FileJson().write(test_data, test_file)
    read_data = file_io.FileJson().read(test_file)
    
    # Then it should be equal to the original test_data
    assert test_data == read_data

    # Remove test files and directory
    shutil.rmtree(os.path.dirname(test_file))

    
def test_FileMat_interface():
    # Given the following directory and dictionary data
    test_file, test_data = generate_test_file_and_data()
    
    # When perform write it to file and read
    file_io.FileMat().write(test_data, test_file)
    read_data = file_io.FileMat().read(test_file)
    shutil.rmtree(os.path.dirname(test_file))

    read_data = {key_: value_ for key_, value_ in zip(read_data.keys(), read_data.values()) if key_[:2] != "__" and key_[-2:] != "__"}
    
    # Then it should be equal to the original test_data
    assert test_data.keys() == read_data.keys()
    assert test_data.values() == [val[0][0] for val in read_data.values()]

if __name__ == "__main__":
    import nose
    nose.run()    

    
    
    
    
    
    
