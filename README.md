# slum-orei
Semantic segmentation of slum areas from high-resolution RGB images using three different architectures: [U-Net](https://arxiv.org/abs/1505.04597), [FPN](https://arxiv.org/abs/1612.03144), and [Linknet](https://arxiv.org/abs/1707.03718), implemented in the [segmentation_models](https://github.com/qubvel/segmentation_models). Slum prediction implementation using [Smoothly-Blend-Image-Patches](https://github.com/Vooban/Smoothly-Blend-Image-Patches).

This repo provides slum dataset in Jakarta and Medan, all required modules, and scripts to run the model.

## Requirements
- [segmentation_models](https://github.com/qubvel/segmentation_models)
- [keras](https://keras.io/) == 2.10 or [tensorflow](https://www.tensorflow.org/) == 2.10
- [numpy](https://numpy.org/)
- [rasterio](https://rasterio.readthedocs.io/en/latest/)
- [matplotlib](https://matplotlib.org/)
- [sklearn](https://scikit-learn.org/stable/)
- [patchify](https://pypi.org/project/patchify/)
- [albumentations](https://albumentations.ai/)
- [Smoothly-Blend-Image-Patches](https://github.com/Vooban/Smoothly-Blend-Image-Patches)

## Segmentation Models installation
### PyPI stable package
```
pip install -U segmentation-models
```

### PyPI latest package
```
pip install -U --pre segmentation-models
```

### Source latest version
```
pip install git+https://github.com/qubvel/segmentation_models
```

## Semantic segmentation of slum areas
### Image patching
```
python image_patching.py
```

### Split train/val data
```
python split_train_val.py
```

### Train/val
```
python train_val_slum.py
```

### Slum prediction
```
python predict_slum.py
```

## Results
| Area      | Val IoU | Val F1 | 
| :---:     | :---:   | :---:  | 
| Jakarta   | 64.63   | 70.44  |
| Medan     | 33.77   | 41.35  |

More detailed description and results can be found in the [project report].
