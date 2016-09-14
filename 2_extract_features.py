#-*- coding: utf-8 -*-

import object_detector.file_io as file_io
import object_detector.factory as factory

DEFAULT_CONFIG_FILE = "conf/car_side.json"
PATCH_SIZE = (32, 96)

if __name__ == "__main__":
    conf = file_io.FileJson().read(DEFAULT_CONFIG_FILE)
     
    # 1. Build FeatureExtrator instance
    extractor = factory.Factory.create_extractor(conf["descriptor"]["algorithm"], conf["descriptor"]["parameters"], PATCH_SIZE)
      
    # 2. Get Feature sets
    extractor.add_positive_sets(image_dir=conf["dataset"]["pos_data_dir"],
                             pattern=conf["dataset"]["pos_format"], 
                             annotation_path=conf["dataset"]['annotations_dir'],
                             sample_ratio=conf["extractor"]["sampling_ratio_for_positive_images"],
                             padding=conf["extractor"]['padding'],
                             )
     
    # Todo : positive sample 숫자에 따라 negative sample 숫자를 자동으로 정할 수 있도록 설정
    extractor.add_negative_sets(image_dir=conf["dataset"]["neg_data_dir"],
                             pattern=conf["dataset"]["neg_format"],
                             n_samples_per_img=conf["extractor"]["num_patches_per_negative_image"],
                             sample_ratio=conf["extractor"]["sampling_ratio_for_negative_images"])
      
    extractor.summary()
      
    # 3. Save dataset
    extractor.save(data_file=conf["extractor"]["output_file"])
#     del extractor
#       
#     # 4. Test Loading dataset
#     extractor = factory.Factory.create_extractor(conf["descriptor"]["algorithm"], conf["descriptor"]["parameters"], PATCH_SIZE, conf["extractor"]["output_file"])
#     extractor.summary()
 
 
     
