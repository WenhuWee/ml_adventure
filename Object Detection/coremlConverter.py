# import coremltools
# coreml_model = coremltools.converters.keras.convert('./keras-yolo3-master/model_data/yolo.h5')
# coreml_model.save('yolov3.mlmodel')


import tfcoreml as tf_converter

tf_converter.convert( tf_model_path = './model/frozen_inference_graph.pb', mlmodel_path = './model/my_model.mlmodel', output_feature_names = ['num_detections:0', 'detection_boxes:0', 'detection_scores:0', 'detection_classes:0'])	

# bazel-bin/tensorflow/tools/graph_transforms/summarize_graph --in_graph=/Users/Lee/Documents/Code/ml_adventure/ObjectDetection/model/frozen_inference_graph.pb