# Augmentation meom:
# train dataset: 設定したaugmentationが実行される
# valid dataset: augmentationは実行されない
# test dataset: TTA(=Test Time Augmentations)を行う場合には，train datasetと同様のaugmentationが実行される
# TTAとは，test dataに対して，augmentationを行い，そのそれぞれに対してpredictをし，そのaverageの結果を最終的なpredict結果とすること

import albumentations as A
from albumentations.pytorch import ToTensor

def get_augmentations_transform(crop_size=128, p=0.5, phase="train"):
    imagenet_stats = {'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]}
    if phase == "train" or "test":
        aug_factor_list = [
            A.RandomResizedCrop(height=crop_size, width=crop_size, scale=(0.8, 1.0)),
            A.Cutout(num_holes=8, p=p),
            A.RandomRotate90(p=p),
            A.HorizontalFlip(p=p),
            A.VerticalFlip(p=p),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=50),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=p),
            A.OneOf([
                A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(),
            ], p=p),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=p),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
                A.IAAPiecewiseAffine(p=0.3),
            ], p=p),
            ToTensor(normalize=imagenet_stats)
        ]
        transformed_image = A.Compose(aug_factor_list)
        return transformed_image
    elif phase == "valid":
        transformed_image = A.Compose([ToTensor(normalize=imagenet_stats)])
        return transformed_image
    else:
        TypeError("Invalid phase type.")
