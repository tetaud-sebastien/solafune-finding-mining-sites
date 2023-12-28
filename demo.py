import argparse
import os
import torch

import numpy as np
import cv2
import glob
from torch import nn
import pandas as pd
from collections import OrderedDict
import segmentation_models_pytorch as smp
from loguru import logger
import rioxarray as xr

import xarray as xar
import tifffile
import natsort
import warnings
warnings.filterwarnings("ignore")


def normalize(band):
    band_min, band_max = (band.min(), band.max())
    return ((band-band_min)/((band_max - band_min)))

def image_preprocessing(image_path):

    xr_image = xr.open_rasterio(image_path, masked=False).values
    red = xr_image[3,:,:]
    green = xr_image[2,:,:]
    blue = xr_image[1,:,:]
    red_n = normalize(red)
    green_n = normalize(green)
    blue_n = normalize(blue)
    rgb_composite_n= np.dstack((red_n, green_n, blue_n))

    return rgb_composite_n

def load_model(model_path, device):
    """
    Load the trained model from the given path.

    Args:
        model_path (str): Path of the trained model.

    Returns:
        model (nn.Module): The loaded model.

    """
    model = smp.Unet(encoder_name="resnet18", encoder_weights=None,in_channels=3, classes=1, activation='sigmoid')

    if device.type == 'cpu':
        checkpoint = torch.load(model_path, map_location=device)
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:] #remove 'module'
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        model.eval()
        logger.info(model.eval())

    else:
        
        logger.info("==> Loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
        model.cuda()
    
    return model


def main(args):
    """
    Main function to test the trained model on the given test data.

    Args:
        config (dict): The configuration dictionary for the test.
        args (argparse.Namespace): The command-line arguments.

    """
    
    model_path = args.checkpoint_path
    hardware = args.hardware
   
    if hardware=='cuda':
        device = torch.device(hardware)
        model = load_model(model_path, device)
        model.to(hardware)
        if len(str(hardware))>1:
            model= nn.DataParallel(model)
        print("------")
        print("Model run on: {}".format(hardware))
        print("------")

    else:
        device = torch.device(hardware)
        model = load_model(model_path, device)
        model.to(hardware)
        print("------")
        print("Model run on: {}".format(hardware))
        print("------")

    
    # Read input
    input = "/home/sebastien/Documents/projects/solafune-solar-panel-detection/data/evaluation"
    output = "/home/sebastien/Documents/projects/solafune-solar-panel-detection/data/submit"

    if not os.path.isdir(output):
        logger.info("Create submit directory to tif files...")
        os.mkdir(output)

    prediction_dir = "inference_pred"
    if not os.path.exists("inference_pred"):
        logger.info("Create prediction directory to save png...")
        os.makedirs(prediction_dir, exist_ok=True)

    images_path = natsort.natsorted(glob.glob(os.path.join(input, "*.tif"), recursive=False))
    eval_masks = natsort.natsorted(glob.glob(os.path.join("/home/sebastien/Documents/projects/solafune-solar-panel-detection/data/sample", "*.tif"), recursive=False))

    for i in range(len(images_path)):

        path_image = images_path[i]
        logger.info(path_image)
        image = image_preprocessing(path_image)
        
        h, w, _ = image.shape

        image = cv2.resize(image, (32, 32),interpolation=cv2.INTER_NEAREST)/255.0
        image = np.transpose(image, (2,1,0))
        image = torch.Tensor(image)        
        image = torch.unsqueeze(image,dim=0)
        
        with torch.no_grad():

            mask_pred = model(image)

        mask_pred = mask_pred.detach().cpu().numpy()

        mask_pred = mask_pred[0,:,:,:]

        binary_predictions = (mask_pred > 0.5).astype(np.uint8)
        binary_predictions = binary_predictions[0,:,:]


        eval_masks_path = eval_masks[i]
        mask = tifffile.imread(eval_masks_path).astype(np.float64)
        # mask_sample = xr.open_rasterio(eval_masks_path, mask=True)
        shape_mask = mask.shape
        logger.warning(mask.shape)
        logger.warning(binary_predictions.shape)
        binary_predictions = cv2.resize(binary_predictions,(shape_mask[1], shape_mask[0]),interpolation=cv2.INTER_NEAREST)
        logger.warning(binary_predictions.shape)

        head, tail = os.path.split(path_image)
        tail = tail.replace("_s2_","_")
        filename_tif = os.path.join(output,tail.replace("image","mask"))
        tifffile.imwrite(filename_tif, binary_predictions)

        pred_rgb = np.zeros((binary_predictions.shape[0], binary_predictions.shape[1], 3), dtype=np.uint8)
        pred_rgb[..., 0] = binary_predictions * 255  
        pred_rgb[..., 1] = binary_predictions * 255  
        pred_rgb[..., 2] = binary_predictions * 255
        pred_rgb = cv2.resize(pred_rgb,(w,h),interpolation=cv2.INTER_NEAREST)
        
        
        binary_predictions = torch.Tensor(pred_rgb[:,:,0])/255.0
        binary_predictions = binary_predictions.to(torch.int)
        binary_predictions = torch.unsqueeze(binary_predictions,dim=0)
        binary_predictions = binary_predictions.detach().cpu().numpy()
        
        
        

        # x = mask_sample['x'].values
        # y = mask_sample['y'].values
        # band = mask_sample['band'].values

        
        
        # final = xar.DataArray(binary_predictions, dims=('y', 'x'), coords={'y': np.arange(0.5,binary_predictions.shape[1],1), 'x': np.arange(0.5,binary_predictions.shape[2])})
        # print(binary_predictions.shape)
        # mask_sample.values = binary_predictions
        # # Write the xarray DataArray to a TIF file
        # final.rio.to_raster(filename_tif)

        

        image = image[0,:,:,:]

        image = torch.permute(image, (2, 1, 0))
        image = image.detach().cpu().numpy()
        # Scale image values to the range [0, 255]
        image = ((image - image.min()) / (image.max() - image.min())) * 255
        image = image.astype(np.uint8)
        image = cv2.resize(image,(w,h),interpolation=cv2.INTER_NEAREST)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_to_save = np.hstack((image, pred_rgb))
        
        filename = os.path.join(prediction_dir, f'evaluation_mask_{i}.png')
        cv2.imwrite(filename, image_to_save)
       
    logger.info("Done")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluation data generation', fromfile_prefix_chars='@')

    # Input
    parser.add_argument('-c', '--checkpoint_path',type=str,   help='path to a checkpoint to load', default='')
    parser.add_argument("-d", "--hardware", type=str, help="device - gpu/cpu", default='cuda')
    args = parser.parse_args()
    
    main(args)
