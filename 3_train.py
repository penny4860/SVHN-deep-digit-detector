#-*- coding: utf-8 -*-

import object_detector.file_io as file_io
import object_detector.factory as factory

HARD_NEGATIVE_OPTION = True
CONFIGURATION_FILE = "conf/new_format.json"
PATCH_SIZE = (32, 96)

if __name__ == "__main__":
    conf = file_io.FileJson().read(CONFIGURATION_FILE)
    
    #1. Load Features and Labels
    getter = factory.Factory.create_extractor(conf["descriptor"]["algorithm"], conf["descriptor"]["parameters"], PATCH_SIZE, conf["extractor"]["output_file"])
    getter.summary()
    data = getter.get_dataset(include_hard_negative=HARD_NEGATIVE_OPTION)
    
    y = data[:, 0]
    X = data[:, 1:]
 
    #2. Load classifier and Train
    cls = factory.Factory.create_classifier(conf["classifier"]["algorithm"], conf["classifier"]["parameters"])
    
    cls.train(X, y)
    print "[INFO] Training result is as follows"
    print cls.evaluate(X, y)
#     [FeatureGetter INFO] Positive samples: 246, Negative samples: 1925
#     [FeatureGetter INFO] Feature Dimension: 5796
#     [INFO] Training result is as follows
#                  precision    recall  f1-score   support
#     
#             0.0       1.00      1.00      1.00      1925
#             1.0       1.00      0.98      0.99       246
#     
#     avg / total       1.00      1.00      1.00      2171
 
 
    #3. Save classifier
    cls.dump(conf["classifier"]["output_file"])


