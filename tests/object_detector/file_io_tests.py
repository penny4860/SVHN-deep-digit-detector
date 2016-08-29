
import os
import object_detector.file_io as file_io
import shutil
import tempfile

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
    # Given the files
    file_io.FileJson().write()
    file_io.FileJson().read()
    
def test_FileMat_interface():
    # Given the files
    file_io.FileJson().write()
    file_io.FileJson().read()
    
def test_FileHDF5_interface():
    # Given the files
    file_io.FileJson().write()
    file_io.FileJson().read()

if __name__ == "__main__":
    import nose
    nose.run()    

    
    
    
    
    
    
