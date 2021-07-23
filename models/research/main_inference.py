import io
import os
import scipy.misc
import numpy as np
import six
import time
import glob
import pathlib

from six import BytesIO

from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf
from models.research.object_detection.utils import ops as utils_ops
from models.research.object_detection.utils import label_map_util
from models.research.object_detection.utils import visualization_utils as vis_util

IMG_RESULTS_PATH = '/Users/ccampos/Desktop/image_results/'

output_directory = 'inference_graph'
train_record_path = "models/research/object_detection/pv-learning/train.record"
test_record_path = "models/research/object_detection/pv-learning/test.record"
labelmap_path = "models/research/object_detection/pv-learning/labelmap.pbtxt"
base_config_path = "models/research/object_detection/configs/tf2/ssd_efficientdet_d0_512x512_coco17_tpu-8.config"

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

category_index = \
    label_map_util.create_category_index_from_labelmap(labelmap_path, use_display_name=True)

tf.keras.backend.clear_session()
model = tf.saved_model.load(f'models/research/object_detection/{output_directory}/saved_model')


def possible_overlaps(input_dict):
    for idx in range(0, len(input_dict['detection_classes'])):
        for jdx in range(idx, len(input_dict['detection_classes'])):
            if idx == jdx:
                pass
            else:
                box1 = input_dict['detection_boxes'][idx]
                box2 = input_dict['detection_boxes'][jdx]

                if over_lapping_area(box1, box2) > 0:
                    print("over-lapped")
                    return True
    return False


def delete_over_lapping(input_dict):
    output_dict = {"num_detections": input_dict['num_detections'],
                   'detection_classes': np.copy(input_dict['detection_classes']),
                   'detection_boxes': np.copy(input_dict['detection_boxes']),
                   'detection_scores': np.copy(input_dict['detection_scores'])
                   }
    # while possible_overlaps(output_dict):
    delete_idx = list()
    for idx in range(0, len(input_dict['detection_classes'])):
        for jdx in range(idx, len(input_dict['detection_classes'])):
            if idx == jdx or (idx in delete_idx and jdx in delete_idx):
                pass
            else:
                box1 = output_dict['detection_boxes'][idx]
                box2 = output_dict['detection_boxes'][jdx]
                if over_lapping_area(box1, box2) > 0:
                    if get_area(box1) > get_area(box2):
                        detection_classes = output_dict['detection_classes'][idx]
                        detection_scores = output_dict['detection_scores'][idx]
                    else:
                        detection_classes = output_dict['detection_classes'][jdx]
                        detection_scores = output_dict['detection_scores'][jdx]
                    p1 = [max(box1[0], box2[0]), max(box1[2], box2[2])]
                    p2 = [max(box1[1], box2[1]), max(box1[3], box2[3])]
                    new_coords = [p1[0], p1[1], p2[0], p2[1]]
                    detection_boxes = np.ndarray(shape=(4,), dtype=np.float32, buffer=np.array(new_coords))
                    np.append(output_dict['detection_boxes'], detection_boxes)
                    np.append(output_dict['detection_classes'], detection_classes)
                    np.append(output_dict['detection_scores'], detection_scores)
                    if jdx not in delete_idx:
                        delete_idx.append(jdx)
                    if idx not in delete_idx:
                        delete_idx.append(idx)
                    output_dict['num_detections'] += 1
    delete_idx.sort(reverse=True)
    for idx in delete_idx:
        np.delete(output_dict['detection_boxes'], idx)
        np.delete(output_dict['detection_classes'], idx)
        np.delete(output_dict['detection_scores'], idx)
    output_dict['num_detections'] -= len(delete_idx)
    print(output_dict['num_detections'])
    return output_dict


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: a file path (this can be local or on colossus)

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data)).convert('RGB')
    (im_width, im_height) = image.size
    np_array = np.array(image.getdata())
    reshaped = np_array.reshape((im_height, im_width, 3))
    return reshaped.astype(np.uint8)


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict


def get_image_name(img_abs_path):
    parts = img_abs_path.split("/")
    return parts[-1]


def over_lapping_area(box1, box2):
    area_1 = get_area(box1)
    area_2 = get_area(box2)
    x_dist = min(box1[2], box2[2]) - max(box1[0], box2[0])
    y_dist = min(box1[3], box2[3]) - max(box1[1], box2[1])
    intersecting_area = 0
    if x_dist > 0 and y_dist > 0:
        intersecting_area = x_dist * y_dist
    return area_1 + area_2 - intersecting_area


def get_area(box):
    return abs(box[0] - box[2]) * abs(box[1] - box[3])


def overlaps(box1, box2):

    if box1[0] >= box2[2] or box1[2] >= box2[0]:
        return False

    if box1[3] >= box2[1] or box2[3] >= box1[1]:
        return False

    return True

#
# category_index = \
#     label_map_util.create_category_index_from_labelmap(labelmap_path, use_display_name=True)
#
# tf.keras.backend.clear_session()
# model = tf.saved_model.load(f'object_detection/{output_directory}/saved_model')
#
# for image_path in glob.glob('object_detection/pv-learning/images/test/*.png'):
#     image_np = load_image_into_numpy_array(image_path)
#     output_dict = run_inference_for_single_image(model, image_np)
#     vis_util.visualize_boxes_and_labels_on_image_array(
#         image_np,
#         output_dict['detection_boxes'],
#         output_dict['detection_classes'],
#         output_dict['detection_scores'],
#         category_index,
#         instance_masks=output_dict.get('detection_masks_reframed', None),
#         use_normalized_coordinates=True,
#         line_thickness=8)
#     img = Image.fromarray(image_np)
#     img_name = get_image_name(image_path)
#     img.save(IMG_RESULTS_PATH + img_name)
#     label_stats = [[output_dict['detection_scores'][idx],
#                     output_dict['detection_classes'][idx], output_dict['detection_boxes'][idx]]
#                    for idx in range(0, len(output_dict['detection_scores']))]
#     # label_stats = filter(score_filter, label_stats)
#     # for item_lst in label_stats:
#     #     print(item_lst)
#     print(output_dict.keys())
#     print("{} inference made".format(img_name))
