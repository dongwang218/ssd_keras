"""Generate positive and negative bboxes used for training"""
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

aspect_ratios = [0.5, 1.0, 2.0, 3.0] # The list of aspect ratios for the anchor boxes
pos_iou_threshold=0.6
neg_iot_threshold=0.1
testing_iou_threshold=0.01
include_thresh = 0.8
model_name = 'ssd7_dummy'
model_file_pattern = '../../data/corrections/ssd7_dummy_weights_epoch{epoch:02d}_loss{loss:.4f}.h5'
batch_size = 16

#img_height = 300 # Height of the input images
#img_width = 256 # Width of the input images
#min_scale = 0.1 # The scaling factor for the smallest anchor boxes
#max_scale = 0.6 # The scaling factor for the largest anchor boxes
#aug_scale=(0.75, 1, 0.5)
#aug_brightness=(0.5, 2, 0.5)

img_height = 850#640 # Height of the input images
img_width = 640 # Width of the input images
min_scale = 0.02 # The scaling factor for the smallest anchor boxes
max_scale = 0.4 # The scaling factor for the largest anchor boxes
aug_scale=False
aug_brightness=(0.8, 1.2, 0.5)

img_channels = 3 # Number of color channels of the input images
n_classes = 2 # Number of classes including the background class
scales = None # An explicit list of anchor box scaling factors. If this is passed, it will override `min_scale` and `max_scale`.
two_boxes_for_ar1 = True # Whether or not you want to generate two anchor boxes for aspect ratio 1
limit_boxes = False # Whether or not you want to limit the anchor boxes to lie entirely within the image boundaries
variances = [1.0, 1.0, 1.0, 1.0] # The list of variances by which the encoded target coordinates are scaled
coords = 'centroids' # Whether the box coordinates to be used should be in the 'centroids' or 'minmax' format, see documentation
normalize_coords = False # Whether or not the model is supposed to use relative coordinates that are within [0,1]


# 2: Build the Keras model (and possibly load some trained weights)

K.clear_session() # Clear previous models from memory.
# The output `predictor_sizes` is needed below to set up `SSDBoxEncoder`
model, predictor_sizes = build_model(image_size=(img_height, img_width, img_channels),
                                     n_classes=n_classes,
                                     min_scale=min_scale,
                                     max_scale=max_scale,
                                     scales=scales,
                                     aspect_ratios_global=aspect_ratios,
                                     aspect_ratios_per_layer=None,
                                     two_boxes_for_ar1=two_boxes_for_ar1,
                                     limit_boxes=limit_boxes,
                                     variances=variances,
                                     coords=coords,
                                     normalize_coords=normalize_coords)
model.load_weights('../../data/corrections/ssd7_6_weights.h5')
#model = load_model('./ssd7_0.h5')

### Set up training

# 3: Instantiate an Adam optimizer and the SSD loss function and compile the model

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-05)

ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

# 4: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function

ssd_box_encoder = SSDBoxEncoder(img_height=img_height,
                                img_width=img_width,
                                n_classes=n_classes,
                                predictor_sizes=predictor_sizes,
                                min_scale=min_scale,
                                max_scale=max_scale,
                                scales=scales,
                                aspect_ratios_global=aspect_ratios,
                                aspect_ratios_per_layer=None,
                                two_boxes_for_ar1=two_boxes_for_ar1,
                                limit_boxes=limit_boxes,
                                variances=variances,
                                pos_iou_threshold=pos_iou_threshold,
                                neg_iot_threshold=neg_iot_threshold,
                                coords=coords,
                                normalize_coords=normalize_coords)

# 5: Create the training set batch generator

# TODO: Set the paths to your datasets here.

# Training dataset
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

val_dataset.parse_csv(images_path=val_images_path,
                      labels_path=val_labels_path,
                      input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                      include_classes='all')

val_generator = val_dataset.generate(batch_size=batch_size,
                                     train=True,
                                     ssd_box_encoder=ssd_box_encoder,
                                     equalize=False,
                                     brightness=False,
                                     flip=False,
                                     translate=False,
                                     scale=False,
                                     max_crop_and_resize=False,
                                     full_crop_and_resize=False,
                                     random_crop=(img_height, img_width, 1, 20, False),
                                     crop=False,
                                     resize=False,
                                     gray=False,
                                     limit_boxes=True,
                                     include_thresh=0.8,
                                     diagnostics=False)

n_val_samples = val_dataset.get_n_samples()

# Change the online data augmentation settings as you like
train_generator = train_dataset.generate(batch_size=batch_size,
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

print('n_train_samples', n_train_samples)

epochs = 1
model.optimizer = Adam(lr=0.00001)

from keras.callbacks import Callback
import shutil

class CollectTrainingBbox(Callback):
  def __init__(self, gen, loss_func, working_dir):
    self.gen = gen
    self.loss_func = loss_func
    self.counter = 0
    self.working_dir = working_dir
    self.training_log = None

  def on_train_begin(self, logs):
    print('generating training log jpg and xml')
    try:
      shutil.rmtree(self.working_dir)
    except:
      pass
    if not os.path.exists(self.working_dir):
      os.makedirs(self.working_dir)

    self.training_log = open(os.path.join(self.working_dir, 'training_log.xml'), 'w')
    self.training_log.write("""<?xml version='1.0' encoding='ISO-8859-1'?>
<?xml-stylesheet type='text/xsl' href='image_metadata_stylesheet.xsl'?>
<dataset>
<name>imglab dataset</name>
<comment>Created by imglab tool.</comment>
<images>
""")


  def on_batch_begin(self, dummy1, dummy2):
    last_batch = self.gen.last_batch
    batch_X = last_batch['X']
    this_filenames = last_batch['filenames']
    y_true_plain = last_batch['y_true_plain']
    y_true = last_batch['y_true']

    y_pred = model.predict_on_batch(batch_X)
    loss, positives, negatives = self.loss_func.compute_loss(y_true, y_pred, diagnosis = True)
    val = np.mean(K.eval(loss))
    print('pre_train_loss', val)

    positives = K.eval(positives)
    negatives = K.eval(negatives)
    assert(len(this_filenames) == y_pred.shape[0])
    # ********* debug begin **********
    #print('generated %s in batch' % len(this_filenames))

    for i in range(len(this_filenames)):
      new_filename = os.path.basename(this_filenames[i]).split('.jpg')[0] + '-%s.jpg' % self.counter
      self.counter += 1
      im = Image.fromarray(batch_X[i])
      width, height = im.size
      im.save(os.path.join(self.working_dir, new_filename))
      self.training_log.write("<image file='%s'>\n" % new_filename)
      for index in np.where(positives[i, :] > 0)[0]:
        box = y_true_plain[i, index, :]
        assert(box[1] == 1)
        cx, cy, w, h = box[2:6].astype(np.int32)
        self.training_log.write("  <box top='%s' left='%s' width='%s' height='%s'>\n    <label>pos</label>\n  </box>" %
                           (cy-h//2, cx - w//2, w, h))
      for index in np.where(negatives[i, :] > 0)[0]:
        box = y_true_plain[i, index, :]
        assert(box[0] == 1)
        cx, cy, w, h = box[2:6].astype(np.int32)
        self.training_log.write("  <box top='%s' left='%s' width='%s' height='%s'>\n    <label>neg</label>\n  </box>" %
                           (cy-h//2, cx - w//2, w, h))
      self.training_log.write("</image>\n")
  # ********* debug end ************

  def on_train_end(self, logs):
    self.training_log.write("</images>\n</dataset>\n")
    self.trainong_log.close()

history = model.fit_generator(generator = train_generator,
                              steps_per_epoch = ceil(n_train_samples/batch_size),
                              epochs = epochs,
                              callbacks = [
                                  ModelCheckpoint(model_file_pattern,
                                                           monitor='val_loss',
                                                           verbose=1,
                                                           save_best_only=True,
                                                           save_weights_only=True,
                                                           mode='auto',
                                                           period=1),
                                           #EarlyStopping(monitor='val_loss',
                                           #              min_delta=0.001,
                                           #              patience=2),
                                           ReduceLROnPlateau(monitor='val_loss',
                                                             factor=0.8,
                                                             patience=5,
                                                             epsilon=0.001,
                                                             verbose=1,
                                                             cooldown=10),
                                  CollectTrainingBbox(train_dataset, ssd_loss,
                                                      working_dir = '/host/data/corrections/generated_training')],
                              validation_data = val_generator,
                              validation_steps = ceil(n_val_samples/batch_size))
