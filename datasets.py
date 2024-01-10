import numpy as np
import cv2
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils
from loguru import logger
import rioxarray as xr
from PIL import Image
from albumentations.pytorch import ToTensorV2
import albumentations as A
import torchvision.transforms.functional as F

import warnings
warnings.filterwarnings("ignore")


def center_crop(img, dim):
    """
    Center crop an image.
    This function crops an image by taking a rectangular region from the center of the image. 
    The size of the rectangular region is determined by the input dimensions.
    Args:
        img (ndarray): The image to be cropped. The image should be a 2D numpy array.
        dim (tuple): A tuple of integers representing the width and height of the crop window.
    Returns:
        ndarray: The center-cropped image as a 2D numpy array.
    Example:
        >>> import numpy as np
        >>> img = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
        >>> cropped_img = center_crop(img, (50, 50))
    """
    width, height = img.shape[1], img.shape[0]
    # process crop width and height for max available dimension
    crop_width = dim[0] if dim[0] < img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0]
    mid_x, mid_y = int(width / 2), int(height / 2)
    cw2, ch2 = int(crop_width / 2), int(crop_height / 2)
    crop_img = img[mid_y - ch2:mid_y + ch2, mid_x - cw2:mid_x + cw2]
    return np.ascontiguousarray(crop_img)


def get_random_crop(image, mask, crop_width=256, crop_height=256):
    """
    Get a random crop from the image and mask.

    Args:
        image (numpy.ndarray): The input image (numpy array).
        mask (numpy.ndarray): The input mask (numpy array).
        crop_width (int): Width of the crop (default is 256).
        crop_height (int): Height of the crop (default is 256).

    Returns:
        tuple: A tuple containing the cropped image and mask as numpy arrays.
    """
    max_x = mask.shape[1] - crop_width
    max_y = mask.shape[0] - crop_height
    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)
    crop_mask = mask[y: y + crop_height, x: x + crop_width]
    crop_image = image[y: y + crop_height, x: x + crop_width, :]
    return crop_image, crop_mask


def data_augmentation(image):
    
    image = np.array(image)
    apply_transform = ["yes", "no"]
    
    if random.choice(apply_transform)=="yes":
        
        
        # Define your augmentation transformations
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),  # Random horizontal flip with a probability of 0.5
            A.VerticalFlip(p=0.5),    # Random vertical flip with a probability of 0.5
            # Add more augmentations as needed
        ])

        # Perform augmentation on the image
        augmented = transform(image=image)
        # Retrieve the augmented image
        augmented_image = augmented['image']
        # augmented_image = np.transpose(augmented_image,(2,1,0))
        
    else:
        augmented_image = image
        
    return augmented_image


def normalize(band):
    band_min, band_max = (band.min(), band.max())
    return ((band-band_min)/((band_max - band_min)))


def image_preprocessing(image_path):

    image = xr.open_rasterio(image_path, masked=False).values
    red = image[3,:,:]
    green = image[2,:,:]
    blue = image[1,:,:]
    red_n = normalize(red)
    green_n = normalize(green)
    blue_n = normalize(blue)
    rgb_composite_n= np.dstack((red_n, green_n, blue_n))
    
    
    return rgb_composite_n


class TrainDataset(Dataset):
    """
    Custom training dataset class.
    """

    def __init__(self, df_path, normalize, data_augmentation):
        """
        Initialize the training dataset.

        Args:
            df_path (DataFrame): A DataFrame containing file paths and dataset information.
            transforms (callable): A function/transform to apply to the data (default is None).
        """
        self.df_path = df_path
        self.normalize = normalize
        self.data_augmentation = data_augmentation
    

    def __getitem__(self, index):

        img_path = self.df_path.image_path.iloc[index]
        image = image_preprocessing(img_path)

        if self.data_augmentation: 
            image = data_augmentation(image)
        
        if self.normalize:
            transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        else:
            transform = transforms.Compose([transforms.ToTensor()])

        image = transform(image)
        
        target = self.df_path.target.iloc[index]
        
        return image, target

    def __len__(self):
        return len(self.df_path)


class EvalDataset(Dataset):


    def __init__(self, df_path, normalize):
        
        self.df_path = df_path
        self.normalize = normalize

    def __getitem__(self, index):

        img_path = self.df_path.image_path.iloc[index]
        image = image_preprocessing(img_path)

        if self.normalize:

            transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            
        else:
            transform = transforms.Compose([transforms.ToTensor()])

        image = transform(image)
        # image = self.transform(image)


        # image = np.transpose(image, (2,1,0))
        # image = torch.Tensor(image)        

        target = self.df_path.target.iloc[index] 

        return image, target


    def __len__(self):
        return len(self.df_path, )


class TestDataset(Dataset):


    def __init__(self, df_path, normalize):
        
        self.df_path = df_path
        self.normalize = normalize


    def __getitem__(self, index):

        img_path = self.df_path.image_path.iloc[index]
        image = image_preprocessing(img_path)

        if self.normalize:

            transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            
        else:
            transform = transforms.Compose([transforms.ToTensor()])
            

        image = transform(image)

        # image = np.transpose(image, (2,1,0))
        # image = torch.Tensor(image)
    
    
        return image

    def __len__(self):
        return len(self.df_path)

    