
import numpy as np
import os
import sys
import tensorflow as tf

import matplotlib.pyplot as plt
from PIL import Image
from timeit import default_timer as timer

from object_detection.utils import ops as utils_ops

# if tf.__version__ < '1.4.0':
#     raise ImportError(
#         'Please upgrade your tensorflow installation to v1.4.* or later!')

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# MODEL_NAME = 'ssdlite_mobilenet_v2_coco'
# PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
# PATH_TO_LABELS = MODEL_NAME + '/mscoco_label_map.pbtxt'
# NUM_CLASSES = 90

PATH_TO_FROZEN_GRAPH = './checkpoint/save_17958/frozen_inference_graph.pb'
PATH_TO_LABELS = './data/pascal_label_map.pbtxt'
NUM_CLASSES = 1

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = './test_images'
TEST_IMAGE_PATHS = [os.path.join(
    PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 2)]
print(TEST_IMAGE_PATHS)

# Size, in inches, of the output images.
# IMAGE_SIZE = (450, 300)

def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {
                output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(
                    tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(
                    tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(
                    tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [
                                           real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                                           real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict, feed_dict={
                                   image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(
                output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

if True:
    import cv2
    video_path = './keras-yolo3-master/1_720.mp4'
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()

    isOutput = False
    if isOutput:
        out = cv2.VideoWriter('./out.mp4', video_FourCC, video_fps, video_size)
    
    count = 0

    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image_np = load_image_into_numpy_array(image)

        if True:
        # if curr_fps == 12 :
            image_np_expanded = np.expand_dims(image_np, axis=0)
            output_dict = run_inference_for_single_image(image_np, detection_graph)
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks'),
                use_normalized_coordinates=True,
                line_thickness=7,
                min_score_thresh=0.3)

            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time

        count = count + 1
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0

        result = image_np

        red, green, blue = result.T 
        result = np.array([blue, green, red])
        result = result.transpose()

        im = Image.fromarray(result).convert('RGB')
        im.save("./out_images/out" + str(count) + ".jpeg")

        # cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.50, color=(255, 0, 0), thickness=2)
        # cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        # cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

if  False:
    # image = Image.open(image_path)
    image = Image.open('./test_images/image5.jpg')
    image_np = load_image_into_numpy_array(image)

    image_np_expanded = np.expand_dims(image_np, axis=0)

    output_dict = run_inference_for_single_image(image_np, detection_graph)
    # print(output_dict['detection_scores'])
    vis_util.visualize_boxes_and_labels_on_image_array(image_np,
                                                       output_dict['detection_boxes'],
                                                       output_dict['detection_classes'],
                                                       output_dict['detection_scores'],
                                                       category_index,
                                                       instance_masks=output_dict.get(
                                                           'detection_masks'),
                                                       use_normalized_coordinates=True,
                                                       line_thickness=8,
                                                       min_score_thresh=0.1)

    # plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(image_np)
    # plt.imshow(image)
    plt.show()

# floyd run --gpu --env tensorflow-1.9 --data tigerfloyd/datasets/tf_lebronjames/1:/data --data tigerfloyd/datasets/sdd_mobilenet/1:/model --data tigerfloyd/datasets/tf_ob/3:tf_ob --tensorboard 'bash ./run.sh'
