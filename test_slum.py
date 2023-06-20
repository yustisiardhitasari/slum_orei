import os
import numpy as np
import keras
import segmentation_models as sm
import utils

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# Environment setting
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, '../segmentation_models/')
DATA_DIR = os.path.join(ROOT_DIR, 'data/Jakarta/')
ckpts_dir = os.path.join(ROOT_DIR, 'ckpts/Jakarta/')

x_test_dir = os.path.join(DATA_DIR, 'test')
y_test_dir = os.path.join(DATA_DIR, 'testannot')

# Set parameters
MODEL = 'fpn' # option: ['unet', 'fpn', 'linknet']
BACKBONE = 'vgg16'
LR = 0.0001
CHANNELS = 3

# Dataset for test images
test_dataset = utils.Dataset(
    x_test_dir, 
    y_test_dir, 
    classes=['slum'],
)

test_dataloader = utils.Dataloder(test_dataset, batch_size=1, shuffle=False)

# Load best weights
if MODEL == 'unet':
    model = sm.Unet(backbone_name=BACKBONE, input_shape=(None, None, CHANNELS), classes=1, activation='sigmoid')
elif MODEL == 'fpn':
    model = sm.FPN(backbone_name=BACKBONE, input_shape=(None, None, CHANNELS), classes=1, activation='sigmoid')
elif MODEL == 'linknet':
    model = sm.Linknet(backbone_name=BACKBONE, input_shape=(None, None, CHANNELS), classes=1, activation='sigmoid')
else:
    raise Exception("Models not found")

model.compile(
    optimizer = keras.optimizers.Adam(LR),
    loss=sm.losses.binary_focal_dice_loss,
    metrics=[sm.metrics.iou_score, sm.metrics.f1_score, sm.metrics.precision, sm.metrics.recall])
model.load_weights(ckpts_dir+'best_'+MODEL+"_"+BACKBONE+'_model.h5')

scores = model.evaluate_generator(test_dataloader)

metrics = [sm.metrics.iou_score, sm.metrics.f1_score, sm.metrics.precision, sm.metrics.recall]
print("Loss: {:.5}".format(scores[0]))
for metric, value in zip(metrics, scores[1:]):
    print("mean {}: {:.5}".format(metric.__name__, value))

# Plot
n = 5
ids = np.random.choice(np.arange(len(test_dataset)), size=n)

for i in ids:
    image, gt_mask = test_dataset[i]
    image = np.expand_dims(image, axis=0)
    pr_mask = model.predict(image).round()
    
    utils.visualize(
        image=utils.denormalize(image.squeeze()),
        gt_mask=gt_mask[..., 0].squeeze(),
        pr_mask=pr_mask[..., 0].squeeze(),
    )
