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



def data_augmentation(image, verbose=False):

    optical_transform = A.Compose([
    A.HorizontalFlip(p=random.uniform(0, 1)),
    A.VerticalFlip(p=random.uniform(0, 1)),
    A.RandomBrightnessContrast(p=random.uniform(0, 1)),
    A.RandomGamma(p=random.uniform(0, 1)),
    # A.GaussNoise(p=random.uniform(0, 1)),
    A.ColorJitter(p=random.uniform(0, 1))])
    
    apply_transform = ["yes", "no"]
    if random.choice(apply_transform)=="yes":
        augmented = optical_transform(image=image)
        aug_image = augmented['image']

        transform_info = optical_transform.get_dict_with_id()
        
        # print("Applied transformations:")
        # for key, value in transform_info.items():
        #     print(f"{key}: {value}")
    else:
        aug_image = image
        
    if verbose:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(image)
        ax1.set_title("Image")
        ax1.axis('off')
        ax2.imshow(aug_image)
        ax2.set_title("Augmented Image")
        ax2.axis('off')
    return aug_image




# def data_augmentation(image):
    
#     image = np.array(image)
#     apply_transform = ["yes", "no"]
    
#     if random.choice(apply_transform)=="yes":
        
        
#         # Define your augmentation transformations
#         transform = A.Compose([
#             A.HorizontalFlip(p=0.5),  # Random horizontal flip with a probability of 0.5
#             A.VerticalFlip(p=0.5),    # Random vertical flip with a probability of 0.5
#             # Add more augmentations as needed
#         ])

#         # Perform augmentation on the image
#         augmented = transform(image=image)
#         # Retrieve the augmented image
#         augmented_image = augmented['image']
#         # augmented_image = np.transpose(augmented_image,(2,1,0))
        
#     else:
#         augmented_image = image
        
#     return augmented_image


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



def image_preprocessing_index(image_path):

    image = xr.open_rasterio(image_path, masked=False).values
    
    nwdi = (image[2,:,:]-image[7,:,:])/(image[2,:,:]+image[7,:,:])
    # nwdi = normalize(nwdi)

    ndvi = (image[7,:,:]-image[3,:,:])/(image[7,:,:]+image[3,:,:])
    # ndvi = normalize(ndvi)

    msi = image[10,:,:]/image[7,:,:]
    # msi = normalize(msi)

    image_index = np.dstack((ndvi, nwdi, msi))
    
    image_index = normalize(image_index)
    print(image_index.shape)
    # image_index= np.transpose(image_index, (1, 2, 0))
    return image_index


def image_preprocessing_pca(image_path):


    from sklearn.decomposition import PCA

    image = xr.open_rasterio(image_path, masked=False).values
    reshaped_data = image.reshape((12, -1)).T  # Transpose for the correct shape
    n_components = 3  # Number of principal components
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(reshaped_data)
    
    pca_result_reshaped = pca_result.T.reshape((n_components, 512, 512))
    # pca_result_reshaped = normalize(pca_result_reshaped)
    pca_result_reshaped = np.transpose(pca_result_reshaped, (1, 2, 0))
    return pca_result_reshaped


class TrainDataset(Dataset):
    """
    Custom training dataset class.
    """
    # def __init__(self, df_path, normalize, data_augmentation):
    def __init__(self, df_path, normalize, preprocessing, resize, data_augmentation):
        """
        Initialize the training dataset.

        Args:
            df_path (DataFrame): A DataFrame containing file paths and dataset information.
            transforms (callable): A function/transform to apply to the data (default is None).
        """
        self.df_path = df_path
        self.normalize = normalize
        self.data_augmentation = data_augmentation
        self.preprocessing = preprocessing
        self.resize = resize

    

    def __getitem__(self, index):

        img_path = self.df_path.image_path.iloc[index]
        # image = image_prepreprocessing(img_path)

        if self.preprocessing == "RGB":
            image = image_preprocessing(image_path=img_path)
        elif self.preprocessing == "INDEX":
            image = image_preprocessing_index(image_path=img_path)
        elif self.preprocessing == "PCA":
            image = image_preprocessing_pca(image_path=img_path)

        if self.data_augmentation: 
            image = data_augmentation(image)
        
        if self.normalize:
            transform = transforms.Compose([
            transforms.ToTensor(),

            # Custom Normalization
            transforms.Normalize(mean=[0.13799362, 0.12012463, 0.09399955], 
                                 std=[0.038435966, 0.030695442, 0.027098808])]),

            # Imagnet Normalization
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        else:
            transform = transforms.Compose([transforms.ToTensor(),transforms.Resize(self.resize)])

        image = transform(image)
        
       
        
        target = self.df_path.target.iloc[index]
        
        return image, target

    def __len__(self):
        return len(self.df_path)


class EvalDataset(Dataset):

    # def __init__(self, df_path, , normalize):
    def __init__(self, df_path, preprocessing, resize,normalize):
        
        self.df_path = df_path
        self.normalize = normalize
        self.preprocessing = preprocessing
        self.resize = resize

    def __getitem__(self, index):

        img_path = self.df_path.image_path.iloc[index]
        # image = image_prepreprocessing(img_path)


        if self.preprocessing == "RGB":
            image = image_preprocessing(image_path=img_path)
        elif self.preprocessing == "INDEX":
            image = image_preprocessing_index(image_path=img_path)
        elif self.preprocessing == "PCA":
            image = image_preprocessing_pca(image_path=img_path)

        if self.normalize:

            transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            
        else:
            transform = transforms.Compose([transforms.ToTensor(),transforms.Resize(self.resize)])

        image = transform(image)
        
        target = self.df_path.target.iloc[index] 

        return image, target


    def __len__(self):
        return len(self.df_path, )


class TestDataset(Dataset):

    def __init__(self, df_path, preprocessing, resize, normalize):
    # def __init__(self, df_path, normalize):
        
        self.df_path = df_path
        self.normalize = normalize
        self.preprocessing = preprocessing
        self.resize = resize


    def __getitem__(self, index):

        img_path = self.df_path.image_path.iloc[index]
        # image = image_prepreprocessing(img_path)

        if self.preprocessing == "RGB":
            image = image_preprocessing(image_path=img_path)
        elif self.preprocessing == "INDEX":
            image = image_preprocessing_index(image_path=img_path)
        elif self.preprocessing == "PCA":
            image = image_preprocessing_pca(image_path=img_path)

        if self.normalize:

            transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            
        else:
            transform = transforms.Compose([transforms.ToTensor(),transforms.Resize(self.resize)])
            

        image = transform(image)    
    
        return image

    def __len__(self):
        return len(self.df_path)

    