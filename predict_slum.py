import os
import rasterio
import numpy as np
import keras
import segmentation_models as sm
from sklearn.preprocessing import MinMaxScaler
from patchify import patchify, unpatchify
from smooth_tiled_predictions import predict_img_with_smooth_windowing

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# Environment setting
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, '../segmentation_models/')
DATA_DIR = os.path.join(ROOT_DIR, 'data/Jakarta/')
ckpts_dir = os.path.join(ROOT_DIR, 'ckpts/Jakarta/')

input_image = os.path.join(DATA_DIR, 'image/'+'Jakarta2aa_1m.tif')
input_label = os.path.join(DATA_DIR, 'annot/'+'Jakarta2aa_1m.tif')
output_label = os.path.join(DATA_DIR, 'pred/'+'Jakarta2aa_1m_pred.tif')

# Load model
MODEL = 'fpn' # option: ['unet', 'fpn', 'linknet']
BACKBONE = 'vgg16'
LR = 0.0001
patch_size = 512
channels = 3

# Load best weights
if MODEL == 'unet':
    model = sm.Unet(backbone_name=BACKBONE, input_shape=(None, None, channels), classes=1, activation='sigmoid')
elif MODEL == 'fpn':
    model = sm.FPN(backbone_name=BACKBONE, input_shape=(None, None, channels), classes=1, activation='sigmoid')
elif MODEL == 'linknet':
    model = sm.Linknet(backbone_name=BACKBONE, input_shape=(None, None, channels), classes=1, activation='sigmoid')
else:
    raise Exception("Models not found")

model.compile(
    optimizer = keras.optimizers.Adam(LR),
    loss=sm.losses.binary_focal_dice_loss,
    metrics=[sm.metrics.iou_score, sm.metrics.f1_score, sm.metrics.precision, sm.metrics.recall])
model.load_weights(ckpts_dir+'best_'+MODEL+"_"+BACKBONE+'_model.h5')


# Read image for prediction
with rasterio.open(input_image) as src:
    profile = src.profile
    image = src.read()
    image = np.moveaxis(image, 0, -1)
    # Scale image
    image = MinMaxScaler().fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape)
    dims = image.shape


# Predict slum
predictions_smooth = predict_img_with_smooth_windowing(
    image,
    window_size=patch_size,
    subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
    nb_classes=1,
    pred_func=(
        lambda img_batch_subdiv: model.predict(img_batch_subdiv)
    )
)
final_prediction = np.argmax(predictions_smooth, axis=2)

# Save as tif
profile.update(
    dtype = rasterio.uint8,
    count = 1
)
with rasterio.open(output_label, 'w', **profile) as dst:
    dst.write(final_prediction, indexes=1)