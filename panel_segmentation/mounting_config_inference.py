import io
import os
import scipy.misc
import numpy as np
import six
import time
import glob
import pathlib
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont, ImageColor
import tensorflow as tf
import collections
import string_int_label_map_pb2
from google.protobuf import text_format

IMG_RESULTS_PATH = '/Users/ccampos/Desktop/image_results/'

"""
This script is adapted from the Tensorflow models/research/ folder.
"""

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def _validate_label_map(label_map):
  """Checks if a label map is valid.

  Args:
    label_map: StringIntLabelMap to validate.

  Raises:
    ValueError: if label map is invalid.
  """
  for item in label_map.item:
    if item.id < 0:
      raise ValueError('Label map ids should be >= 0.')
    if (item.id == 0 and item.name != 'background' and
        item.display_name != 'background'):
      raise ValueError('Label map id 0 is reserved for the background label')


def load_labelmap(path):
  """Loads label map proto.

  Args:
    path: path to StringIntLabelMap proto text file.
  Returns:
    a StringIntLabelMapProto
  """
  with tf.io.gfile.GFile(path, 'r') as fid:
    label_map_string = fid.read()
    label_map = string_int_label_map_pb2.StringIntLabelMap()
    try:
      text_format.Merge(label_map_string, label_map)
    except text_format.ParseError:
      label_map.ParseFromString(label_map_string)
  _validate_label_map(label_map)
  return label_map


def create_category_index(categories):
  """Creates dictionary of COCO compatible categories keyed by category id.

  Args:
    categories: a list of dicts, each of which has the following keys:
      'id': (required) an integer id uniquely identifying this category.
      'name': (required) string representing category name
        e.g., 'cat', 'dog', 'pizza'.

  Returns:
    category_index: a dict containing the same entries as categories, but keyed
      by the 'id' field of each category.
  """
  category_index = {}
  for cat in categories:
    category_index[cat['id']] = cat
  return category_index


def convert_label_map_to_categories(label_map,
                                    max_num_classes,
                                    use_display_name=True):
  """Given label map proto returns categories list compatible with eval.

  This function converts label map proto and returns a list of dicts, each of
  which  has the following keys:
    'id': (required) an integer id uniquely identifying this category.
    'name': (required) string representing category name
      e.g., 'cat', 'dog', 'pizza'.
    'keypoints': (optional) a dictionary of keypoint string 'label' to integer
      'id'.
  We only allow class into the list if its id-label_id_offset is
  between 0 (inclusive) and max_num_classes (exclusive).
  If there are several items mapping to the same id in the label map,
  we will only keep the first one in the categories list.

  Args:
    label_map: a StringIntLabelMapProto or None.  If None, a default categories
      list is created with max_num_classes categories.
    max_num_classes: maximum number of (consecutive) label indices to include.
    use_display_name: (boolean) choose whether to load 'display_name' field as
      category name.  If False or if the display_name field does not exist, uses
      'name' field as category names instead.

  Returns:
    categories: a list of dictionaries representing all possible categories.
  """
  categories = []
  list_of_ids_already_added = []
  if not label_map:
    label_id_offset = 1
    for class_id in range(max_num_classes):
      categories.append({
          'id': class_id + label_id_offset,
          'name': 'category_{}'.format(class_id + label_id_offset)
      })
    return categories
  for item in label_map.item:
    if not 0 < item.id <= max_num_classes:
      logging.info(
          'Ignore item %d since it falls outside of requested '
          'label range.', item.id)
      continue
    if use_display_name and item.HasField('display_name'):
      name = item.display_name
    else:
      name = item.name
    if item.id not in list_of_ids_already_added:
      list_of_ids_already_added.append(item.id)
      category = {'id': item.id, 'name': name}
      if item.HasField('frequency'):
        if item.frequency == string_int_label_map_pb2.LVISFrequency.Value(
            'FREQUENT'):
          category['frequency'] = 'f'
        elif item.frequency == string_int_label_map_pb2.LVISFrequency.Value(
            'COMMON'):
          category['frequency'] = 'c'
        elif item.frequency == string_int_label_map_pb2.LVISFrequency.Value(
            'RARE'):
          category['frequency'] = 'r'
      if item.HasField('instance_count'):
        category['instance_count'] = item.instance_count
      if item.keypoints:
        keypoints = {}
        list_of_keypoint_ids = []
        for kv in item.keypoints:
          if kv.id in list_of_keypoint_ids:
            raise ValueError('Duplicate keypoint ids are not allowed. '
                             'Found {} more than once'.format(kv.id))
          keypoints[kv.label] = kv.id
          list_of_keypoint_ids.append(kv.id)
        category['keypoints'] = keypoints
      categories.append(category)
  return categories


def create_categories_from_labelmap(label_map_path, use_display_name=True):
  """Reads a label map and returns categories list compatible with eval.

  This function converts label map proto and returns a list of dicts, each of
  which  has the following keys:
    'id': an integer id uniquely identifying this category.
    'name': string representing category name e.g., 'cat', 'dog'.
    'keypoints': a dictionary of keypoint string label to integer id. It is only
      returned when available in label map proto.

  Args:
    label_map_path: Path to `StringIntLabelMap` proto text file.
    use_display_name: (boolean) choose whether to load 'display_name' field
      as category name.  If False or if the display_name field does not exist,
      uses 'name' field as category names instead.

  Returns:
    categories: a list of dictionaries representing all possible categories.
  """
  label_map = load_labelmap(label_map_path)
  max_num_classes = max(item.id for item in label_map.item)
  return convert_label_map_to_categories(label_map, max_num_classes,
                                         use_display_name)


def create_category_index_from_labelmap(label_map_path, use_display_name=True):
  """Reads a label map and returns a category index.

  Args:
    label_map_path: Path to `StringIntLabelMap` proto text file.
    use_display_name: (boolean) choose whether to load 'display_name' field
      as category name.  If False or if the display_name field does not exist,
      uses 'name' field as category names instead.

  Returns:
    A category index, which is a dictionary that maps integer ids to dicts
    containing categories, e.g.
    {1: {'id': 1, 'name': 'dog'}, 2: {'id': 2, 'name': 'cat'}, ...}
  """
  categories = create_categories_from_labelmap(label_map_path, use_display_name)
  return create_category_index(categories)


def draw_bounding_box_on_image_array(image,
                                     ymin,
                                     xmin,
                                     ymax,
                                     xmax,
                                     color='red',
                                     thickness=4,
                                     display_str_list=(),
                                     use_normalized_coordinates=True):
  """Adds a bounding box to an image (numpy array).

  Bounding box coordinates can be specified in either absolute (pixel) or
  normalized coordinates by setting the use_normalized_coordinates argument.

  Args:
    image: a numpy array with shape [height, width, 3].
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color,
                             thickness, display_str_list,
                             use_normalized_coordinates)
  np.copyto(image, np.array(image_pil))

  
def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='red',
                               thickness=4,
                               display_str_list=(),
                               use_normalized_coordinates=True):
  """Adds a bounding box to an image.

  Bounding box coordinates can be specified in either absolute (pixel) or
  normalized coordinates by setting the use_normalized_coordinates argument.

  Each string in display_str_list is displayed on a separate line above the
  bounding box in black text on a rectangle filled with the input 'color'.
  If the top of the bounding box extends to the edge of the image, the strings
  are displayed below the bounding box.

  Args:
    image: a PIL.Image object.
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  if use_normalized_coordinates:
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
  else:
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
  if thickness > 0:
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
               (left, top)],
              width=thickness,
              fill=color)
  try:
    font = ImageFont.truetype('arial.ttf', 24)
  except IOError:
    font = ImageFont.load_default()

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = bottom + total_display_str_height
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle(
        [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                          text_bottom)],
        fill=color)
    draw.text(
        (left + margin, text_bottom - text_height - margin),
        display_str,
        fill='black',
        font=font)
    text_bottom -= text_height - 2 * margin


def draw_mask_on_image_array(image, mask, color='red', alpha=0.4):
  """Draws mask on an image.

  Args:
    image: uint8 numpy array with shape (img_height, img_height, 3)
    mask: a uint8 numpy array of shape (img_height, img_height) with
      values between either 0 or 1.
    color: color to draw the keypoints with. Default is red.
    alpha: transparency value between 0 and 1. (default: 0.4)

  Raises:
    ValueError: On incorrect data type for image or masks.
  """
  if image.dtype != np.uint8:
    raise ValueError('`image` not of type np.uint8')
  if mask.dtype != np.uint8:
    raise ValueError('`mask` not of type np.uint8')
  if image.shape[:2] != mask.shape:
    raise ValueError('The image has spatial dimensions %s but the mask has '
                     'dimensions %s' % (image.shape[:2], mask.shape))
  rgb = ImageColor.getrgb(color)
  pil_image = Image.fromarray(image)

  solid_color = np.expand_dims(
      np.ones_like(mask), axis=2) * np.reshape(list(rgb), [1, 1, 3])
  pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert('RGBA')
  pil_mask = Image.fromarray(np.uint8(255.0*alpha*(mask > 0))).convert('L')
  pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
  np.copyto(image, np.array(pil_image.convert('RGB')))
  

def draw_keypoints_on_image_array(image,
                                  keypoints,
                                  keypoint_scores=None,
                                  min_score_thresh=0.5,
                                  color='red',
                                  radius=2,
                                  use_normalized_coordinates=True,
                                  keypoint_edges=None,
                                  keypoint_edge_color='green',
                                  keypoint_edge_width=2):
  """Draws keypoints on an image (numpy array).

  Args:
    image: a numpy array with shape [height, width, 3].
    keypoints: a numpy array with shape [num_keypoints, 2].
    keypoint_scores: a numpy array with shape [num_keypoints]. If provided, only
      those keypoints with a score above score_threshold will be visualized.
    min_score_thresh: A scalar indicating the minimum keypoint score required
      for a keypoint to be visualized. Note that keypoint_scores must be
      provided for this threshold to take effect.
    color: color to draw the keypoints with. Default is red.
    radius: keypoint radius. Default value is 2.
    use_normalized_coordinates: if True (default), treat keypoint values as
      relative to the image.  Otherwise treat them as absolute.
    keypoint_edges: A list of tuples with keypoint indices that specify which
      keypoints should be connected by an edge, e.g. [(0, 1), (2, 4)] draws
      edges from keypoint 0 to 1 and from keypoint 2 to 4.
    keypoint_edge_color: color to draw the keypoint edges with. Default is red.
    keypoint_edge_width: width of the edges drawn between keypoints. Default
      value is 2.
  """
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  draw_keypoints_on_image(image_pil,
                          keypoints,
                          keypoint_scores=keypoint_scores,
                          min_score_thresh=min_score_thresh,
                          color=color,
                          radius=radius,
                          use_normalized_coordinates=use_normalized_coordinates,
                          keypoint_edges=keypoint_edges,
                          keypoint_edge_color=keypoint_edge_color,
                          keypoint_edge_width=keypoint_edge_width)
  np.copyto(image, np.array(image_pil))


def draw_keypoints_on_image(image,
                            keypoints,
                            keypoint_scores=None,
                            min_score_thresh=0.5,
                            color='red',
                            radius=2,
                            use_normalized_coordinates=True,
                            keypoint_edges=None,
                            keypoint_edge_color='green',
                            keypoint_edge_width=2):
  """Draws keypoints on an image.

  Args:
    image: a PIL.Image object.
    keypoints: a numpy array with shape [num_keypoints, 2].
    keypoint_scores: a numpy array with shape [num_keypoints].
    min_score_thresh: a score threshold for visualizing keypoints. Only used if
      keypoint_scores is provided.
    color: color to draw the keypoints with. Default is red.
    radius: keypoint radius. Default value is 2.
    use_normalized_coordinates: if True (default), treat keypoint values as
      relative to the image.  Otherwise treat them as absolute.
    keypoint_edges: A list of tuples with keypoint indices that specify which
      keypoints should be connected by an edge, e.g. [(0, 1), (2, 4)] draws
      edges from keypoint 0 to 1 and from keypoint 2 to 4.
    keypoint_edge_color: color to draw the keypoint edges with. Default is red.
    keypoint_edge_width: width of the edges drawn between keypoints. Default
      value is 2.
  """
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  keypoints = np.array(keypoints)
  keypoints_x = [k[1] for k in keypoints]
  keypoints_y = [k[0] for k in keypoints]
  if use_normalized_coordinates:
    keypoints_x = tuple([im_width * x for x in keypoints_x])
    keypoints_y = tuple([im_height * y for y in keypoints_y])
  if keypoint_scores is not None:
    keypoint_scores = np.array(keypoint_scores)
    valid_kpt = np.greater(keypoint_scores, min_score_thresh)
  else:
    valid_kpt = np.where(np.any(np.isnan(keypoints), axis=1),
                         np.zeros_like(keypoints[:, 0]),
                         np.ones_like(keypoints[:, 0]))
  valid_kpt = [v for v in valid_kpt]

  for keypoint_x, keypoint_y, valid in zip(keypoints_x, keypoints_y, valid_kpt):
    if valid:
      draw.ellipse([(keypoint_x - radius, keypoint_y - radius),
                    (keypoint_x + radius, keypoint_y + radius)],
                   outline=color, fill=color)
  if keypoint_edges is not None:
    for keypoint_start, keypoint_end in keypoint_edges:
      if (keypoint_start < 0 or keypoint_start >= len(keypoints) or
          keypoint_end < 0 or keypoint_end >= len(keypoints)):
        continue
      if not (valid_kpt[keypoint_start] and valid_kpt[keypoint_end]):
        continue
      edge_coordinates = [
          keypoints_x[keypoint_start], keypoints_y[keypoint_start],
          keypoints_x[keypoint_end], keypoints_y[keypoint_end]
      ]
      draw.line(
          edge_coordinates, fill=keypoint_edge_color, width=keypoint_edge_width)



def _get_multiplier_for_color_randomness():
  """Returns a multiplier to get semi-random colors from successive indices.

  This function computes a prime number, p, in the range [2, 17] that:
  - is closest to len(STANDARD_COLORS) / 10
  - does not divide len(STANDARD_COLORS)

  If no prime numbers in that range satisfy the constraints, p is returned as 1.

  Once p is established, it can be used as a multiplier to select
  non-consecutive colors from STANDARD_COLORS:
  colors = [(p * i) % len(STANDARD_COLORS) for i in range(20)]
  """
  num_colors = len(STANDARD_COLORS)
  prime_candidates = [5, 7, 11, 13, 17]

  # Remove all prime candidates that divide the number of colors.
  prime_candidates = [p for p in prime_candidates if num_colors % p]
  if not prime_candidates:
    return 1

  # Return the closest prime number to num_colors / 10.
  abs_distance = [np.abs(num_colors / 10. - p) for p in prime_candidates]
  num_candidates = len(abs_distance)
  inds = [i for _, i in sorted(zip(abs_distance, range(num_candidates)))]
  return prime_candidates[inds[0]]



def reframe_box_masks_to_image_masks(box_masks, boxes, image_height,
                                     image_width, resize_method='bilinear'):
  """Transforms the box masks back to full image masks.

  Embeds masks in bounding boxes of larger masks whose shapes correspond to
  image shape.

  Args:
    box_masks: A tensor of size [num_masks, mask_height, mask_width].
    boxes: A tf.float32 tensor of size [num_masks, 4] containing the box
           corners. Row i contains [ymin, xmin, ymax, xmax] of the box
           corresponding to mask i. Note that the box corners are in
           normalized coordinates.
    image_height: Image height. The output mask will have the same height as
                  the image height.
    image_width: Image width. The output mask will have the same width as the
                 image width.
    resize_method: The resize method, either 'bilinear' or 'nearest'. Note that
      'bilinear' is only respected if box_masks is a float.

  Returns:
    A tensor of size [num_masks, image_height, image_width] with the same dtype
    as `box_masks`.
  """
  resize_method = 'nearest' if box_masks.dtype == tf.uint8 else resize_method
  # TODO(rathodv): Make this a public function.
  def reframe_box_masks_to_image_masks_default():
    """The default function when there are more than 0 box masks."""
    def transform_boxes_relative_to_boxes(boxes, reference_boxes):
      boxes = tf.reshape(boxes, [-1, 2, 2])
      min_corner = tf.expand_dims(reference_boxes[:, 0:2], 1)
      max_corner = tf.expand_dims(reference_boxes[:, 2:4], 1)
      denom = max_corner - min_corner
      # Prevent a divide by zero.
      denom = tf.math.maximum(denom, 1e-4)
      transformed_boxes = (boxes - min_corner) / denom
      return tf.reshape(transformed_boxes, [-1, 4])

    box_masks_expanded = tf.expand_dims(box_masks, axis=3)
    num_boxes = tf.shape(box_masks_expanded)[0]
    unit_boxes = tf.concat(
        [tf.zeros([num_boxes, 2]), tf.ones([num_boxes, 2])], axis=1)
    reverse_boxes = transform_boxes_relative_to_boxes(unit_boxes, boxes)

    # TODO(vighneshb) Use matmul_crop_and_resize so that the output shape
    # is static. This will help us run and test on TPUs.
    resized_crops = tf.image.crop_and_resize(
        image=box_masks_expanded,
        boxes=reverse_boxes,
        box_ind=tf.range(num_boxes),
        crop_size=[image_height, image_width],
        method=resize_method,
        extrapolation_value=0)
    return tf.cast(resized_crops, box_masks.dtype)

  image_masks = tf.cond(
      tf.shape(box_masks)[0] > 0,
      reframe_box_masks_to_image_masks_default,
      lambda: tf.zeros([0, image_height, image_width, 1], box_masks.dtype))
  return tf.squeeze(image_masks, axis=3)


def visualize_boxes_and_labels_on_image_array(
    image,
    boxes,
    classes,
    scores,
    category_index,
    instance_masks=None,
    instance_boundaries=None,
    keypoints=None,
    keypoint_scores=None,
    keypoint_edges=None,
    track_ids=None,
    use_normalized_coordinates=False,
    max_boxes_to_draw=20,
    min_score_thresh=.5,
    agnostic_mode=False,
    line_thickness=4,
    mask_alpha=.4,
    groundtruth_box_visualization_color='black',
    skip_boxes=False,
    skip_scores=False,
    skip_labels=False,
    skip_track_ids=False):
  """Overlay labeled boxes on an image with formatted scores and label names.

  This function groups boxes that correspond to the same location
  and creates a display string for each detection and overlays these
  on the image. Note that this function modifies the image in place, and returns
  that same image.

  Args:
    image: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    instance_masks: a uint8 numpy array of shape [N, image_height, image_width],
      can be None.
    instance_boundaries: a numpy array of shape [N, image_height, image_width]
      with values ranging between 0 and 1, can be None.
    keypoints: a numpy array of shape [N, num_keypoints, 2], can
      be None.
    keypoint_scores: a numpy array of shape [N, num_keypoints], can be None.
    keypoint_edges: A list of tuples with keypoint indices that specify which
      keypoints should be connected by an edge, e.g. [(0, 1), (2, 4)] draws
      edges from keypoint 0 to 1 and from keypoint 2 to 4.
    track_ids: a numpy array of shape [N] with unique track ids. If provided,
      color-coding of boxes will be determined by these ids, and not the class
      indices.
    use_normalized_coordinates: whether boxes is to be interpreted as
      normalized coordinates or not.
    max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
      all boxes.
    min_score_thresh: minimum score threshold for a box or keypoint to be
      visualized.
    agnostic_mode: boolean (default: False) controlling whether to evaluate in
      class-agnostic mode or not.  This mode will display scores but ignore
      classes.
    line_thickness: integer (default: 4) controlling line width of the boxes.
    mask_alpha: transparency value between 0 and 1 (default: 0.4).
    groundtruth_box_visualization_color: box color for visualizing groundtruth
      boxes
    skip_boxes: whether to skip the drawing of bounding boxes.
    skip_scores: whether to skip score when drawing a single detection
    skip_labels: whether to skip label when drawing a single detection
    skip_track_ids: whether to skip track id when drawing a single detection

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
  """
  # Create a display string (and color) for every box location, group any boxes
  # that correspond to the same location.
  box_to_display_str_map = collections.defaultdict(list)
  box_to_color_map = collections.defaultdict(str)
  box_to_instance_masks_map = {}
  box_to_instance_boundaries_map = {}
  box_to_keypoints_map = collections.defaultdict(list)
  box_to_keypoint_scores_map = collections.defaultdict(list)
  box_to_track_ids_map = {}
  if not max_boxes_to_draw:
    max_boxes_to_draw = boxes.shape[0]
  for i in range(boxes.shape[0]):
    if max_boxes_to_draw == len(box_to_color_map):
      break
    if scores is None or scores[i] > min_score_thresh:
      box = tuple(boxes[i].tolist())
      if instance_masks is not None:
        box_to_instance_masks_map[box] = instance_masks[i]
      if instance_boundaries is not None:
        box_to_instance_boundaries_map[box] = instance_boundaries[i]
      if keypoints is not None:
        box_to_keypoints_map[box].extend(keypoints[i])
      if keypoint_scores is not None:
        box_to_keypoint_scores_map[box].extend(keypoint_scores[i])
      if track_ids is not None:
        box_to_track_ids_map[box] = track_ids[i]
      if scores is None:
        box_to_color_map[box] = groundtruth_box_visualization_color
      else:
        display_str = ''
        if not skip_labels:
          if not agnostic_mode:
            if classes[i] in six.viewkeys(category_index):
              class_name = category_index[classes[i]]['name']
            else:
              class_name = 'N/A'
            display_str = str(class_name)
        if not skip_scores:
          if not display_str:
            display_str = '{}%'.format(round(100*scores[i]))
          else:
            display_str = '{}: {}%'.format(display_str, round(100*scores[i]))
        if not skip_track_ids and track_ids is not None:
          if not display_str:
            display_str = 'ID {}'.format(track_ids[i])
          else:
            display_str = '{}: ID {}'.format(display_str, track_ids[i])
        box_to_display_str_map[box].append(display_str)
        if agnostic_mode:
          box_to_color_map[box] = 'DarkOrange'
        elif track_ids is not None:
          prime_multipler = _get_multiplier_for_color_randomness()
          box_to_color_map[box] = STANDARD_COLORS[
              (prime_multipler * track_ids[i]) % len(STANDARD_COLORS)]
        else:
          box_to_color_map[box] = STANDARD_COLORS[
              classes[i] % len(STANDARD_COLORS)]

  # Draw all boxes onto image.
  for box, color in box_to_color_map.items():
    ymin, xmin, ymax, xmax = box
    if instance_masks is not None:
      draw_mask_on_image_array(
          image,
          box_to_instance_masks_map[box],
          color=color,
          alpha=mask_alpha
      )
    if instance_boundaries is not None:
      draw_mask_on_image_array(
          image,
          box_to_instance_boundaries_map[box],
          color='red',
          alpha=1.0
      )
    draw_bounding_box_on_image_array(
        image,
        ymin,
        xmin,
        ymax,
        xmax,
        color=color,
        thickness=0 if skip_boxes else line_thickness,
        display_str_list=box_to_display_str_map[box],
        use_normalized_coordinates=use_normalized_coordinates)
    if keypoints is not None:
      keypoint_scores_for_box = None
      if box_to_keypoint_scores_map:
        keypoint_scores_for_box = box_to_keypoint_scores_map[box]
      draw_keypoints_on_image_array(
          image,
          box_to_keypoints_map[box],
          keypoint_scores_for_box,
          min_score_thresh=min_score_thresh,
          color=color,
          radius=line_thickness / 2,
          use_normalized_coordinates=use_normalized_coordinates,
          keypoint_edges=keypoint_edges,
          keypoint_edge_color=color,
          keypoint_edge_width=line_thickness // 2)

  return image


def possible_overlaps(input_dict):
    for idx in range(0, len(input_dict['detection_classes'])):
        for jdx in range(idx, len(input_dict['detection_classes'])):
            if idx == jdx:
                pass
            else:
                box1 = input_dict['detection_boxes'][idx]
                box2 = input_dict['detection_boxes'][jdx]
                o_area = overlapping_area(box1, box2)
                if o_area > 0:
                    print("over-lapped: {}".format(o_area))
                    return True
    return False


def delete_overlapping(input_dict):
    delete_idx = list()
    out_dict = {"num_detections": input_dict['num_detections'],
                       'detection_classes': np.copy(input_dict['detection_classes']),
                       'detection_boxes': np.copy(input_dict['detection_boxes']),
                       'detection_scores': np.copy(input_dict['detection_scores'])
                       }
    for idx in range(0, len(input_dict['detection_classes'])):
        for jdx in range(idx, len(input_dict['detection_classes'])):
            if idx == jdx or (idx in delete_idx and jdx in delete_idx):
                pass
            else:
                box1 = out_dict['detection_boxes'][idx]
                box2 = out_dict['detection_boxes'][jdx]
                if overlapping_area(box1, box2) > 0:
                    if get_area(box1) > get_area(box2):
                        detection_classes = out_dict['detection_classes'][idx]
                        detection_scores = out_dict['detection_scores'][idx]
                    else:
                        detection_classes = out_dict['detection_classes'][jdx]
                        detection_scores = out_dict['detection_scores'][jdx]
                    p1 = [min(box1[0], box2[0], box1[2], box2[2]), min(box1[1], box2[1], box1[3], box2[3])]
                    p2 = [max(box1[0], box2[0], box1[2], box2[2]), max(box1[1], box2[1], box1[3], box2[3])]
                    new_coords = [p1[0], p1[1], p2[0], p2[1]]
                    detection_boxes = np.ndarray(shape=(1, 4), dtype=np.float32, buffer=np.array(new_coords))
                    out_dict['detection_boxes'] = np.append(out_dict['detection_boxes'], detection_boxes, axis=0)
                    out_dict['detection_classes'] = np.append(out_dict['detection_classes'], detection_classes)
                    out_dict['detection_scores'] = np.append(out_dict['detection_scores'], detection_scores)
                    if jdx not in delete_idx:
                        delete_idx.append(jdx)
                    if idx not in delete_idx:
                        delete_idx.append(idx)
                    out_dict['num_detections'] += 1
    delete_idx.sort(reverse=True)
    for idx in delete_idx:
        out_dict['detection_scores'] = np.delete(out_dict['detection_scores'], idx)
        out_dict['detection_classes'] = np.delete(out_dict['detection_classes'], idx)
        out_dict['detection_boxes'] = np.delete(out_dict['detection_boxes'], idx, 0)
    out_dict['num_detections'] -= len(delete_idx)
    return out_dict


def filter_score(input_dict, percent):
    total_deleted = 0
    for idx in range(len(input_dict['detection_scores'])-1, -1, -1):
        score = input_dict['detection_scores'][idx]
        if score <= percent:
            input_dict['detection_scores'] = np.delete(input_dict['detection_scores'], idx)
            input_dict['detection_classes'] = np.delete(input_dict['detection_classes'], idx)
            input_dict['detection_boxes'] = np.delete(input_dict['detection_boxes'], idx, 0)
            total_deleted += 1
    input_dict['num_detections'] -= total_deleted
    return input_dict


def delete_overlapping_in_image(input_dict):
    output_dict = delete_overlapping(input_dict)
    while possible_overlaps(output_dict):
        output_dict = delete_overlapping(input_dict=output_dict)
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
        detection_masks_reframed = reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    return output_dict


def get_image_name(img_abs_path):
    parts = img_abs_path.split("/")
    return parts[-1]


def overlapping_area(box1, box2):
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
