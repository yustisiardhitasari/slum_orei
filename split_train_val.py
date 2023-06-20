import os
import numpy as np
import random
import shutil


# Environment setting
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, '../segmentation_models/')
DATA_DIR = os.path.join(ROOT_DIR, 'data/Jakarta/')

train_dir = os.path.join(DATA_DIR, 'train/')
trainannot_dir = os.path.join(DATA_DIR, 'trainannot/')
val_dir = os.path.join(DATA_DIR, 'val/')
valannot_dir = os.path.join(DATA_DIR, 'valannot/')

if not os.path.exists(val_dir):
    os.makedirs(val_dir)
    
if not os.path.exists(valannot_dir):
    os.makedirs(valannot_dir)

# List files inside directory
train_files = os.listdir(train_dir)
trainannot_files = os.listdir(trainannot_dir)

# Generate random number of file indexes
n_val = np.ceil(0.2*len(train_files)).astype(int) # train:val=80:20
randint_val_files = random.sample(range(0, len(train_files)), n_val)

# Move files
for i in randint_val_files:
    train_file = train_files[i]
    shutil.move(os.path.join(train_dir, train_file), os.path.join(val_dir, train_file))

    trainannot_file = trainannot_files[i]
    shutil.move(os.path.join(trainannot_dir, trainannot_file), os.path.join(valannot_dir, trainannot_file))
