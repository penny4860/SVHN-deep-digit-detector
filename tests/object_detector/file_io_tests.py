
import os
import object_detector.file_io as file_io
import shutil

def test_list_files():
    # Given the following directories and files
    TEST_DIR = "TestDirectory"
    
    os.mkdir(TEST_DIR)
    os.mkdir(TEST_DIR + "\\sub1")
    os.mkdir(TEST_DIR + "\\sub2")
    
    files_in_txt = [TEST_DIR+ "\\a.txt", 
                    TEST_DIR+ "\\b.txt", 
                    TEST_DIR+ "\\sub1\\a_in_sub1.txt", 
                    TEST_DIR+ "\\sub2\\b_in_sub2.txt"]
    files_in_log = ["test.log"]
    
    for filename in files_in_txt + files_in_log:
        with open(filename, "w") as _: pass
    
    # When perform crop_bb()
    files_listed_in_txt = file_io.list_files(directory=TEST_DIR, pattern="*.txt", n_files_to_sample=None, recursive_option=True)
    
    # Then it should be have same elements with files_created.
    assert set(files_listed_in_txt) == set(files_in_txt)
    
    # Remove test files and directory
    shutil.rmtree(TEST_DIR)


if __name__ == "__main__":
    import nose
    nose.run()    

    
    
    
    
    
    
