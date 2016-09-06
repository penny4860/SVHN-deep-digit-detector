#-*- coding: utf-8 -*-

import object_detector.extractor as extractor
import object_detector.file_io as file_io
import object_detector.descriptor as descriptor

CONFIGURATION_FILE = "conf/new_format.json"
PATCH_SIZE = (32, 96)

class DescriptorFactory:
    
    def __init__(self):
        pass
    
    @staticmethod
    def create(algorithm, params):
        desc = None
        if algorithm == "hog":
            desc = descriptor.HOG(**params)
            
        if desc is None:
            raise ValueError('Such algorithm is not supported.')

        return desc

if __name__ == "__main__":
    conf = file_io.FileJson().read(CONFIGURATION_FILE)
     
    # 1. Initialize Descriptor instance
    desc = DescriptorFactory.create(conf["descriptor"]["algorithm"], conf["descriptor"]["parameters"])
    
    # 2. Initialize FeatureGetter instance
    getter = extractor.FeatureExtractor(descriptor=desc, patch_size=PATCH_SIZE)
      
    # 3. Get Feature sets
    getter.add_positive_sets(image_dir=conf["dataset"]["pos_data_dir"],
                             pattern=conf["dataset"]["pos_format"], 
                             annotation_path=conf["dataset"]['annotations_dir'],
                             padding=conf["extractor"]['padding'],
                             )
     
    # Todo : positive sample 숫자에 따라 negative sample 숫자를 자동으로 정할 수 있도록 설정
    getter.add_negative_sets(image_dir=conf["dataset"]["neg_data_dir"],
                             pattern=conf["dataset"]["neg_format"],
                             n_samples_per_img=conf["extractor"]["num_patches_per_negative_image"],
                             sample_ratio=conf["extractor"]["sampling_ratio_for_negative_images"])
      
    getter.summary()
      
    # 4. Save dataset
    getter.save(data_file=conf["extractor"]["output_file"])
    del getter
     
    # 5. Test Loading dataset
    getter = extractor.FeatureExtractor.load(descriptor=desc, patch_size=PATCH_SIZE, data_file=conf["extractor"]["output_file"])
    getter.summary()
 
 
     
