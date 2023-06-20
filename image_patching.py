import os
import rasterio
import numpy as np
from patchify import patchify


# Define patch size
patch_size = 512
steps = 256 # patch_size == steps means no overlap

# Environment setting
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, '../segmentation_models/')
DATA_DIR = os.path.join(ROOT_DIR, 'data/Jakarta/')

fp_in = os.path.join(DATA_DIR, 'image/') # change dir to annot/ when patching the annotation image
fp_clip = os.path.join(DATA_DIR, 'image_clipped/') # change dir to annot_clipped/ when patching the annotation image
fp_out = os.path.join(DATA_DIR, 'train/') # change dir to trainannot/ when patching the annotation image

if not os.path.exists(fp_clip):
    os.makedirs(fp_clip)
    
if not os.path.exists(fp_out):
    os.makedirs(fp_out)

# Start patching
print('Start patching ...')
fname = os.listdir(fp_in)
for f, file in enumerate(fname):
    fn = file.rsplit('.', 1)[0]
    
    if file.endswith('.tif'):
        # Open file
        with rasterio.open(fp_in+file) as src:
            # Crop image so that image_shape//patch_size == positive integer
            xsize = src.width//patch_size*patch_size
            ysize = src.height//patch_size*patch_size
            window = rasterio.windows.Window(0, 0, xsize, ysize)
            transform = src.window_transform(window)

            profile = src.profile
            profile.update({
                'width': xsize,
                'height': ysize,
                'transform': transform,
                'dtype': rasterio.uint8,
                'nodata': None,
            })

            # Read image
            arr_img = src.read(window=window) # shape = (channel, y/height/row, x/width/col)

            # Save cropped image as tif
            if not os.path.exists(fp_clip+fn+'_clip.tif'):
                with rasterio.open(fp_clip+fn+'_clip.tif', 'w', **profile) as dst:
                    dst.write(arr_img)
            
            # Rearrange image array so that shape = (y/height/row, x/width/col, channel)
            features = np.moveaxis(arr_img, 0, -1)

            # Image tiling
            patches = patchify(features, (patch_size, patch_size, features.shape[2]), step=steps) #step=patch_size means no overlapping pixels
            patches = np.squeeze(patches, axis=2) # remove unecessary dimension that patchify adds
            print(f'Patches shape: {patches.shape}')

            for i in range(patches.shape[0]):
                for j in range(patches.shape[1]):
                    patch_img = patches[i,j,:,:]

                    # Update window and profile to define the coordinates
                    window = rasterio.windows.Window(j*steps, i*steps, patch_size, patch_size)
                    transform = src.window_transform(window)
                    profile.update({
                        'width': patch_size,
                        'height': patch_size,
                        'transform': transform,
                        'dtype': rasterio.uint8,
                        'nodata': None,
                    })
                    
                    # Rearrange output patches following rasterio indexes
                    arr_out = np.moveaxis(patch_img, -1, 0) # shape = (channel, y/height/row, x/width/col)

                    # Save output patches as tif
                    if not os.path.exists(fp_out+fn+'_patch_'+str(i)+'_'+str(j)+'.tif'):
                        with rasterio.open(fp_out+fn+'_patch_'+str(i)+'_'+str(j)+'.tif', 'w', **profile) as dst:
                            dst.write(arr_out)

print('Done!')