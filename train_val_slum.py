import os
import sys
import time
from datetime import datetime
import keras
import segmentation_models as sm
import matplotlib.pyplot as plt
import utils

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# Environment setting
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, '../segmentation_models/')
DATA_DIR = os.path.join(ROOT_DIR, 'data/Jakarta/')

x_train_dir = os.path.join(DATA_DIR, 'train')
y_train_dir = os.path.join(DATA_DIR, 'trainannot')

x_val_dir = os.path.join(DATA_DIR, 'val')
y_val_dir = os.path.join(DATA_DIR, 'valannot')

ckpts_dir = os.path.join(ROOT_DIR, 'ckpts/Jakarta/')
if not os.path.exists(ckpts_dir):
    os.makedirs(ckpts_dir)


# Set parameters
MODEL = 'fpn' # option: ['unet', 'fpn', 'linknet']
BACKBONE = 'vgg16'
BATCH_SIZE = 2
LR = 0.0001
EPOCHS = 100
ROW, COL, CHANNELS = 512, 512, 3


# Dataset for train images
train_dataset = utils.Dataset(
    x_train_dir,
    y_train_dir,
    classes=['slum'],
    augmentation=utils.get_training_augmentation(),
)
print('Number of training data: {}'.format(len(train_dataset.ids)))

# Dataset for validation images
valid_dataset = utils.Dataset(
    x_val_dir,
    y_val_dir,
    classes=['slum'],
)

train_dataloader = utils.Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = utils.Dataloder(valid_dataset, batch_size=1, shuffle=False)

# Check shapes for errors
assert train_dataloader[0][0].shape == (BATCH_SIZE, ROW, COL, CHANNELS)
assert train_dataloader[0][1].shape == (BATCH_SIZE, ROW, COL, 1)


# Create model
if MODEL == 'unet':
    model = sm.Unet(backbone_name=BACKBONE, input_shape=(None, None, CHANNELS), classes=1, activation='sigmoid')
elif MODEL == 'fpn':
    model = sm.FPN(backbone_name=BACKBONE, input_shape=(None, None, CHANNELS), classes=1, activation='sigmoid')
elif MODEL == 'linknet':
    model = sm.Linknet(backbone_name=BACKBONE, input_shape=(None, None, CHANNELS), classes=1, activation='sigmoid')
else:
    raise Exception("Models not found")

# Define callbacks for learning rate scheduling and best checkpoints saving
callbacks = [
    keras.callbacks.ModelCheckpoint(ckpts_dir+'best_'+MODEL+"_"+BACKBONE+'_model.h5', save_weights_only=True, save_best_only=True, mode='min'),
    keras.callbacks.ReduceLROnPlateau(),
    keras.callbacks.TensorBoard(log_dir=ckpts_dir+'tensorboard_'+MODEL+"_"+BACKBONE+'/'),
]

# Compile keras model with defined optimizer, loss and metrics
model.compile(
    optimizer = keras.optimizers.Adam(LR),
    loss=sm.losses.binary_focal_dice_loss,
    metrics=[sm.metrics.iou_score, sm.metrics.f1_score, sm.metrics.precision, sm.metrics.recall])

model.summary()

# Train model
start_time = time.time()
history = model.fit(
    train_dataloader, 
    steps_per_epoch=len(train_dataloader), 
    epochs=EPOCHS, 
    callbacks=callbacks, 
    validation_data=valid_dataloader, 
    validation_steps=len(valid_dataloader),
)
elapsed_time = time.time() - start_time
print('Training complete. Elapsed time: '+str(elapsed_time))

print('{}-Done!'.format(datetime.now()))


# Plot training & validation iou_score values
plt.figure(figsize=(30, 5))
plt.subplot(121)
plt.plot(history.history['iou_score'])
plt.plot(history.history['val_iou_score'])
plt.title('Model iou_score')
plt.ylabel('iou_score')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig(ckpts_dir+MODEL+"_"+BACKBONE+"_train_val.png", dpi=100)
