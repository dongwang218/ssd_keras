"""Generate bad predictions, so that we can train them separately"""
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
#from matplotlib import pyplot as plt

#%matplotlib inline

from keras_ssd7 import build_model
from keras_ssd_loss import SSDLoss
from ssd_box_encode_decode_utils import SSDBoxEncoder, decode_y, decode_y2
from ssd_batch_generator import BatchGenerator
from keras_layer_AnchorBoxes import AnchorBoxes
from keras_layer_L2Normalization import L2Normalization
from ssd_box_encode_decode_utils import predict_with_patch

import tensorflow as tf

aspect_ratios = [0.5, 1.0, 2.0, 3.0] # The list of aspect ratios for the anchor boxes
pos_iou_threshold=0.6
neg_iot_threshold=0.1
testing_iou_threshold=0.01
include_thresh = 0.8
model_name = 'ssd7_6'
model_file_pattern = '../../data/corrections/ssd7_6_weights_epoch{epoch:02d}_loss{loss:.4f}.h5'
batch_size = 16

scale_up = 1
full_img_height = 1280*scale_up # Height of the input images
full_img_width = 1280*scale_up # Width of the input images
img_channels = 3 # Number of color channels of the input images
n_classes = 2 # Number of classes including the background class
full_min_scale = 0.01 # The scaling factor for the smallest anchor boxes
full_max_scale = 0.2 # The scaling factor for the largest anchor boxes
scales = None # An explicit list of anchor box scaling factors. If this is passed, it will override `min_scale` and `max_scale`.
aspect_ratios = [0.5, 1.0, 2.0, 3.0] # The list of aspect ratios for the anchor boxes
two_boxes_for_ar1 = True # Whether or not you want to generate two anchor boxes for aspect ratio 1
limit_boxes = False # Whether or not you want to limit the anchor boxes to lie entirely within the image boundaries
variances = [1.0, 1.0, 1.0, 1.0] # The list of variances by which the encoded target coordinates are scaled
coords = 'centroids' # Whether the box coordinates to be used should be in the 'centroids' or 'minmax' format, see documentation
normalize_coords = False # Whether or not the model is supposed to use relative coordinates that are within [0,1]

# 2: Build the Keras model (and possibly load some trained weights)

# The output `predictor_sizes` is needed below to set up `SSDBoxEncoder`
full_model, full_predictor_sizes = build_model(image_size=(full_img_height, full_img_width, img_channels),
                                     n_classes=n_classes,
                                     min_scale=full_min_scale,
                                     max_scale=full_max_scale,
                                     scales=scales,
                                     aspect_ratios_global=aspect_ratios,
                                     aspect_ratios_per_layer=None,
                                     two_boxes_for_ar1=two_boxes_for_ar1,
                                     limit_boxes=limit_boxes,
                                     variances=variances,
                                     coords=coords,
                                     normalize_coords=normalize_coords)

full_model.load_weights('../../data/corrections/ssd7_7_weights.h5')

### Make predictions

# 1: Set the generator

train_images_path = '../../data/corrections/set2/sheetimages/'
train_labels_path = '../../data/corrections/math5000_training.csv'

# Validation dataset
val_images_path = '../../data/corrections/set2/sheetimages/'
val_labels_path = '../../data/corrections/math5000_testing.csv'

train_dataset = BatchGenerator(box_output_format=['class_id', 'xmin', 'xmax', 'ymin', 'ymax']) # This is the format in which the generator is supposed to output the labels. At the moment it **must** be the format set here.

train_dataset.parse_csv(images_path=train_images_path,
                        labels_path=train_labels_path,
                        input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'], # This is the order of the first six columns in the CSV file that contains the labels for your dataset. If your labels are in XML format, maybe the XML parser will be helpful, check the documentation.
                        include_classes='all')

# 6: Create the validation set batch generator (if you want to use a validation dataset)

val_dataset = BatchGenerator(box_output_format=['class_id', 'xmin', 'xmax', 'ymin', 'ymax'])

val_images_path = '../../data/corrections/set2/sheetimages/'
val_labels_path = '../../data/corrections/math5000.xml'
val_dataset.parse_dlib_xml(images_path=val_images_path,
                      labels_path=val_labels_path,
                      input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                      include_classes='all')

predict_generator = val_dataset.generate(batch_size=1,
                                         train=False,
                                         equalize=False,
                                         brightness=False,
                                         flip=False,
                                         translate=False,
                                         scale=False,
                                         max_crop_and_resize=False,
                                         full_crop_and_resize=False,
                                         random_crop=(full_img_height//scale_up, full_img_width//scale_up,0,1,True),
                                         crop=False,
                                         resize=(full_img_height,full_img_width),
                                         gray=False,
                                         limit_boxes=True,
                                         include_thresh=0.8,
                                         diagnostics=False)

n_val_samples = val_dataset.get_n_samples()

from ssd_box_encode_decode_utils import iou, iot
from PIL import Image
# bbox: xmin,xmax,ymin,ymax
def match_bbox(pos_iou_threshold, ground_truth, predicted_bbox):
    """
    This match is more loose, as long as there is a overlap it is covered,
    If a predict is not matched to any that way, it is considered false positive.
    """
    true_positive = []
    false_negative = []
    false_positive = []

    positive_boxes = np.zeros((predicted_bbox.shape[0])) # 1 for all negative boxes, 0 otherwise
    for true_box in ground_truth: # For each ground truth box belonging to the current batch item...
        true_box = true_box.astype(np.float)
        if abs(true_box[1] - true_box[0] < 0.001) or abs(true_box[3] - true_box[2] < 0.001):
          continue

        similarities = iou(predicted_bbox, true_box, coords='minmax') # The iou similarities for all anchor boxes
        positive_boxes[similarities >= pos_iou_threshold] = 1
        if np.all(similarities < pos_iou_threshold):
          false_negative.append(true_box)
        else:
          true_positive.append(true_box)
    for index in np.nonzero(positive_boxes == 0)[0]:
      b = predicted_bbox[index]
      if b[1] - b[0] > 0.001:
        false_positive.append(b)
    return true_positive, false_negative, false_positive

result = open('../../data/corrections/set2_testing_positive_negative.xml', 'w')
result.write("""<?xml version='1.0' encoding='ISO-8859-1'?>
<?xml-stylesheet type='text/xsl' href='image_metadata_stylesheet.xsl'?>
<dataset>
<name>imglab dataset</name>
<comment>Created by imglab tool.</comment>
<images>
""")
for i in range(n_val_samples):
    X, y_true, filenames = next(predict_generator)
    #y_pred = full_model.predict(X)
    y_pred = predict_with_patch(full_model, X)

    y_pred_decoded = decode_y2(y_pred,
                           confidence_thresh=0.5,
                          iou_threshold=testing_iou_threshold,
                          top_k='all',
                          input_coords='centroids',
                          normalize_coords=False,
                          img_height=full_img_height,
                          img_width=full_img_width)
    if y_pred_decoded[0].shape[0] > 0:
        y_pred_decoded[0][:, 2:4] /= full_img_width
        y_pred_decoded[0][:, 4:6] /= full_img_height
        predicted = y_pred_decoded[0][:, 2:]
    else:
        predicted = np.array([[0.0,0.0,0.0,0.0]])
    y_true[0] = y_true[0].astype(np.float)
    y_true[0][:, 1:3] /= full_img_width
    y_true[0][:, 3:5] /= full_img_height
    #print(y_true[0][:,1:])
    true_positive, false_negative, false_positive = match_bbox(0.01, y_true[0][:,1:], predicted)
    if len(false_negative) == 0 and len(false_positive) == 0: continue

    img = Image.open(filenames[0], 'r')
    width, height = full_img_width, full_img_height #img.size
    result.write("<image file='%s'>\n" % filenames[0])
    for box in false_negative:
        result.write("  <box top='%s' left='%s' width='%s' height='%s'>\n    <label>neg</label>\n  </box>\n" %
                     (int(box[2] * height), int(box[0] * width),
                      int((box[1]-box[0])*width), int((box[3]-box[2])*height)))
    for box in false_positive:
      result.write("  <box top='%s' left='%s' width='%s' height='%s'>\n    <label>pos</label>\n  </box>\n" %
                     (int(box[2] * height), int(box[0] * width),
                      int((box[1]-box[0])*width), int((box[3]-box[2])*height)))
    result.write("</image>\n")
    result.flush()
result.write("""
</images>
</dataset>""")
result.close()
