import coremltools
coreml_model = coremltools.converters.keras.convert('./keras-yolo3-master/model_data/yolo.h5')
coreml_model.save('yolov3.mlmodel')