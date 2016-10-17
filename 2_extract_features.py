#-*- coding: utf-8 -*-

import object_detector.file_io as file_io
import object_detector.factory as factory
import argparse as ap

DEFAULT_CONFIG_FILE = "conf/svhn.json"

if __name__ == "__main__":
    
    parser = ap.ArgumentParser()
    parser.add_argument('-c', "--config", help="Configuration File", default=DEFAULT_CONFIG_FILE)
    args = vars(parser.parse_args())
    
    conf = file_io.FileJson().read(args["config"])
     
    # 1. Build FeatureExtrator instance
    extractor = factory.Factory.create_extractor(conf["descriptor"]["algorithm"], 
                                                 conf["descriptor"]["parameters"], 
                                                 conf["detector"]["window_dim"])
      
    # 2. Get Feature sets
    extractor.add_positive_sets(annotation_file=conf["dataset"]["annotation_file"],
                             sample_ratio=conf["extractor"]["sampling_ratio_for_positive_images"],
                             padding=conf["extractor"]['padding'],
                             )
    extractor.summary()
     
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
#     extractor = factory.Factory.create_extractor(conf["descriptor"]["algorithm"], conf["descriptor"]["parameters"], conf["detector"]["window_dim"], conf["extractor"]["output_file"])
#     extractor.summary()
 
 
     
