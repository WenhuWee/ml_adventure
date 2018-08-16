from keras.utils import plot_model
from keras.models import load_model

model = load_model('./model_data/yolov3-tiny.h5')
for l in model.layers:
    print (l.output_shape)
# model.summary()
# plot_model(model, to_file='model.png',show_shapes=True)