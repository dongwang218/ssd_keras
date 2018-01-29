"""Generate positive and negative patch images used for training"""
import os
from PIL import Image

import keras
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np


from keras_ssd7 import build_model
from keras_ssd_loss import SSDLoss
from ssd_box_encode_decode_utils import SSDBoxEncoder, decode_y, decode_y2
from ssd_batch_generator import BatchGenerator

import tensorflow as tf


mode = sys.argv[1]
assert(mode in ['train', 'test'])

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

train_images_path = '../../data/corrections/set2/sheetimages/'
train_labels_path = '../../data/corrections/math5000_%sing.xml' % mode
train_dataset.parse_dlib_xml(images_path=train_images_path,
                      labels_path=train_labels_path,
                      input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                      include_classes='all')

predict_generator = train_dataset.generate(batch_size=1,
                                         train=True,
                                         ssd_box_encoder=ssd_box_encoder,
                                         equalize=False,
                                         brightness=aug_brightness, # Randomly change brightness between 0.5 and 2 with probability 0.5
                                         flip=False, # Randomly flip horizontally with probability 0.5
                                         translate=False, #((5, 50), (3, 30), 0.5), # Randomly translate by 5-50 pixels horizontally and 3-30 pixels vertically with probability 0.5
                                         scale=aug_scale, # Randomly scale between 0.75 and 1.3 with probability 0.5
                                         max_crop_and_resize=False,
                                         full_crop_and_resize=False,
                                         random_crop=(img_height, img_width, 1, 20, False),
                                         crop=False,
                                         resize=False,
                                         gray=False,
                                         limit_boxes=True,
                                         include_thresh=0.8,
                                         diagnostics=False)

n_train_samples = train_dataset.get_n_samples()


print('generating training log jpg and xml')
workding_dir = os.path.join('../../data/corrections/patches/%s' % mode)
try:
  shutil.rmtree(working_dir)
except:
  pass
if not os.path.exists(working_dir):
  os.makedirs(self.working_dir)
if not os.path.exists(working_dir + '/0'):
  os.makedirs(self.working_dir + '/0')
if not os.path.exists(working_dir + '/1'):
  os.makedirs(self.working_dir + '/1')

for i in range(n_train_samples):
    X, y_true, filenames = next(predict_generator)
    #y_pred = full_model.predict(X)
    y_pred = full_model.predict(batch_X)
    assert(np.all(np.abs(y_true[:,:, -8:-4] - y_pred[:,:,-8:-4]) < 1e-3))
    #loss, positives, negatives = self.loss_func.compute_loss_nms(y_true, y_pred, diagnosis = True)
    #val = np.mean(K.eval(loss))
    #print('pre_train_loss', val)
    #positives = K.eval(positives)
    #negatives = K.eval(negatives)

    loss, selected = self.loss_func.compute_loss_nms(tf.convert_to_tensor(y_true, dtype=tf.float32),
                                                     tf.convert_to_tensor(y_pred, dtype=tf.float32), diagnosis = True)
    selected = K.eval(selected)

    positives = np.logical_and(selected, np.any(y_true[:, :, 1:-12] > 0, axis=-1))
    negatives = np.logical_and(selected, y_true[:, :, 0] > 0)

    assert(len(filenames) == y_pred.shape[0])

    for i in range(len(filenames)):
      new_filename = os.path.join(train_images_path, os.path.basename(filenames[i]))
      img = Image.read(new_filename)

      for ind, id in enumerate(selected):
        patch_name = new_filename.replace('.jpg', '-%03s.jpg' % ind)
        patch = img.crop(y_true[i, id, -8:-4])
        label = '0' if  y_true[i, id, 0] else '1'
        patch.write(os.path.join(working_dir, label, os.path.basename(patch_name)))
