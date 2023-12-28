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
    model = smp.Unet(encoder_name="resnet50", encoder_weights=None,in_channels=3, classes=1, activation='sigmoid')

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
    input = "/home/sebastien/Documents/projects/solafune-finding-mining-sites/data/evaluation"
    output = "/home/sebastien/Documents/projects/solafune-finding-mining-sites/data/eval_model"
    prediction_dir = "/home/sebastien/Documents/projects/solafune-finding-mining-sites/data/eval_model_png"
    if not os.path.isdir(output):
        logger.info("Create submit directory to tif files...")
        os.mkdir(output)

    
    if not os.path.exists(prediction_dir):
        logger.info("Create prediction directory to save png...")
        os.makedirs(prediction_dir, exist_ok=True)

    df = pd.read_csv("/home/sebastien/Documents/projects/solafune-finding-mining-sites/data_splits/valid_path.csv")

    images_path = df["rgb_path"]
    eval_masks = df["mask_path"]


    iou_metrics = []
    f1_score_metrics = []
    metrics_dict = {}


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
        pred_rgb = np.zeros((binary_predictions.shape[0], binary_predictions.shape[1], 3), dtype=np.uint8)
        pred_rgb[..., 0] = binary_predictions * 255  
        pred_rgb[..., 1] = binary_predictions * 255  
        pred_rgb[..., 2] = binary_predictions * 255
        pred_rgb = cv2.resize(pred_rgb,(w,h),interpolation=cv2.INTER_NEAREST)
        

        mask_gt = xr.open_rasterio(eval_masks[i], masked=True)
        mask_gt = mask_gt.values



        
        binary_predictions = torch.Tensor(pred_rgb[:,:,0])/255.0
        binary_predictions = binary_predictions.to(torch.int)
        binary_predictions = torch.unsqueeze(binary_predictions,dim=0)
        binary_predictions = binary_predictions.detach().cpu().numpy()
        
        eval_masks_path = eval_masks[i]
        mask_sample = xr.open_rasterio(eval_masks_path, mask=True)
        
        from sklearn.metrics import f1_score
        from sklearn.metrics import jaccard_score
        f1_score_metrics.append(f1_score(mask_gt[0,:,:].flatten(), binary_predictions[0,:,:].flatten(),average='binary'))
        iou_metrics.append(jaccard_score(mask_gt[0,:,:].flatten(), binary_predictions[0,:,:].flatten(),average='binary'))
        
        x = mask_sample['x'].values
        y = mask_sample['y'].values
        band = mask_sample['band'].values

        final = xar.DataArray(binary_predictions, dims=('band','y', 'x'), coords={'band': band,'y': np.arange(0.5,binary_predictions.shape[1],1), 'x': np.arange(0.5,binary_predictions.shape[2])})
        head, tail = os.path.split(path_image)
        filename_tif = os.path.join(output,tail.replace("image","mask"))
        final.rio.to_raster(filename_tif)


        mask_gt_rgb = np.zeros((mask_gt.shape[1], mask_gt.shape[2], 3), dtype=np.uint8)
        mask_gt_rgb[..., 0] = mask_gt[0,:,:] * 255
        mask_gt_rgb[..., 1] = mask_gt[0,:,:] * 255
        mask_gt_rgb[..., 2] = mask_gt[0,:,:] * 255

        image = image[0,:,:,:]
        image = torch.permute(image, (2, 1, 0))
        image = image.detach().cpu().numpy()
        # Scale image values to the range [0, 255]
        image = ((image - image.min()) / (image.max() - image.min())) * 255
        image = image.astype(np.uint8)
        image = cv2.resize(image,(w,h),interpolation=cv2.INTER_NEAREST)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_to_save = np.hstack((image, mask_gt_rgb, pred_rgb))
        
        filename = os.path.join(prediction_dir, f'evaluation_mask_{i}.png')
        cv2.imwrite(filename, image_to_save)

    metrics_dict[0] = { "IoU": np.mean(iou_metrics),"F1": np.mean(f1_score_metrics)
                               }
    print(metrics_dict)
       
    logger.info("Done")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluation data generation', fromfile_prefix_chars='@')

    # Input
    parser.add_argument('-c', '--checkpoint_path',type=str,   help='path to a checkpoint to load', default='')
    parser.add_argument("-d", "--hardware", type=str, help="device - gpu/cpu", default='cuda')
    args = parser.parse_args()
    
    main(args)
