import os 
import cv2
import pandas as pd
import torch
import numpy as np

from bokeh.plotting import figure, show
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource
from bokeh.palettes import Category10
from bokeh.plotting import figure, output_file, save

from torch import nn

def dice_coef(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask*groundtruth_mask)
    total_sum = np.sum(pred_mask) + np.sum(groundtruth_mask)
    dice = np.mean(2*intersect/total_sum)
    return round(dice, 3) #round up to 3 decimal places

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, prediction, target):

        prediction = prediction.view(-1)
        target = target.view(-1)
        smooth = 1.0  # Smoothing factor to avoid division by zero
        intersection = torch.sum(prediction * target)
        union = torch.sum(prediction) + torch.sum(target)
        dice = (2.0 * intersection + smooth) / (union + smooth)
        loss = 1.0 - dice  # Ensure the loss remains positive
        
        return loss


class WeightedBCELoss(torch.nn.Module):
    def __init__(self, weight_pos, weight_neg):
        super(WeightedBCELoss, self).__init__()
        self.weight_pos = weight_pos
        self.weight_neg = weight_neg

    def forward(self, inputs, targets):
        # Apply sigmoid to convert inputs to probabilities
        
        
        # Calculate the weighted binary cross-entropy loss
        loss = - (self.weight_pos*targets * torch.log(inputs) + self.weight_neg*(1 - targets) * torch.log(1 - inputs))
        
        # Average the loss over the batch
        loss = torch.mean(loss)
        
        return loss


class BCELoss(torch.nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        
    def forward(self, inputs, targets):
        
        # Calculate the weighted binary cross-entropy loss
        loss = - (targets * torch.log(inputs) + (1 - targets) * torch.log(1 - inputs))
        # Average the loss over the batch
        loss = torch.mean(loss)
        
        return loss


def intersection_over_union(predictions, targets, threshold):
    
    iou_list = []
    binary_predictions = (predictions > threshold).to(torch.uint8)
    
    for i in range(binary_predictions.shape[0]):
        target = targets[i,:,:]
        binary_mask = binary_predictions[i, 0,:,:]
        intersection = torch.logical_and(binary_mask, target)

        union = torch.logical_or(binary_mask, target)
        union_sum = torch.sum(union).item()

        if union_sum == 0:
            iou_list.append(0)
        else:
            iou = torch.sum(intersection).item() / union_sum
            iou_list.append(iou)
    
    ious = torch.tensor(iou_list)
    return ious


def convert_rgb_to_y(img):
    if type(img) == np.ndarray:
        return 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        return 16. + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.
    else:
        raise Exception('Unknown Type', type(img))


def convert_rgb_to_ycbcr(img):
    if type(img) == np.ndarray:
        y = 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
        cb = 128. + (-37.945 * img[:, :, 0] - 74.494 * img[:, :, 1] + 112.439 * img[:, :, 2]) / 256.
        cr = 128. + (112.439 * img[:, :, 0] - 94.154 * img[:, :, 1] - 18.285 * img[:, :, 2]) / 256.
        return np.array([y, cb, cr]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        y = 16. + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.
        cb = 128. + (-37.945 * img[0, :, :] - 74.494 * img[1, :, :] + 112.439 * img[2, :, :]) / 256.
        cr = 128. + (112.439 * img[0, :, :] - 94.154 * img[1, :, :] - 18.285 * img[2, :, :]) / 256.
        return torch.cat([y, cb, cr], 0).permute(1, 2, 0)
    else:
        raise Exception('Unknown Type', type(img))


def convert_ycbcr_to_rgb(img):
    if type(img) == np.ndarray:
        r = 298.082 * img[:, :, 0] / 256. + 408.583 * img[:, :, 2] / 256. - 222.921
        g = 298.082 * img[:, :, 0] / 256. - 100.291 * img[:, :, 1] / 256. - 208.120 * img[:, :, 2] / 256. + 135.576
        b = 298.082 * img[:, :, 0] / 256. + 516.412 * img[:, :, 1] / 256. - 276.836
        return np.array([r, g, b]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        r = 298.082 * img[0, :, :] / 256. + 408.583 * img[2, :, :] / 256. - 222.921
        g = 298.082 * img[0, :, :] / 256. - 100.291 * img[1, :, :] / 256. - 208.120 * img[2, :, :] / 256. + 135.576
        b = 298.082 * img[0, :, :] / 256. + 516.412 * img[1, :, :] / 256. - 276.836
        return torch.cat([r, g, b], 0).permute(1, 2, 0)
    else:
        raise Exception('Unknown Type', type(img))


def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))


def colorize(image, vmin=None, vmax=None, cmap='magma'):

    vmin = image.min() if vmin is None else vmin
    vmax = image.max() if vmax is None else vmax


    if vmin != vmax:
        image = (image - vmin) / (vmax - vmin)
    else:
        image = image * 0.

    image = (image * 255.0).astype(np.uint8)
    image = cv2.applyColorMap(image, cv2.COLORMAP_MAGMA)

    

    return image


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_depth(depth_image, depth_array, filename, save_path):

    """
    Save the depth image and depth array to the specified location.

    Args:
        depth_image (numpy.ndarray): The depth image to be saved.
        depth_array (numpy.ndarray): The depth array to be saved.
        filename (str): The filename of the original image.
        save_path (str): The path to save the depth image and depth array.

    Returns:
        None
    """

    # Extract the directory and filename from the provided path
    directory, filename = os.path.split(filename)

    # Split the filename into name and extension
    name, extension = os.path.splitext(filename)

    # Create the depth filenames
    depth_rgb_filename = f"{name}_depth{extension}"
    depth_array_filename = f"{name}_depth.npy"

    # Construct the full paths for saving the RGB image and depth array
    rgb_path = os.path.join(save_path, 'rgb', depth_rgb_filename)
    depth_path = os.path.join(save_path, 'depth', depth_array_filename)

    # Save the RGB image
    cv2.imwrite(rgb_path, depth_image)

    # Save the depth array in .npy format
    np.save(depth_path, depth_array)
    

def count_model_parameters(model):
    """
    Count number of parameters in a Pytorch Model.

    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_model_size(model):
    size_model = 0
    for param in model.parameters():
        if param.data.is_floating_point():
            size_model += param.numel() * torch.finfo(param.data.dtype).bits
            
        else:
            size_model += param.numel() * torch.iinfo(param.data.dtype).bits
            
    size_model = size_model / 8e6
    print(f"model size: {size_model:.2f} MB")
    return size_model


class Dashboard():

    """
    Generates and saves a dashboard based on a given dataframe.

    Args:
        dataframe (pandas.DataFrame): The dataframe containing the data for the dashboard.

    Attributes:
        dataframe (pandas.DataFrame): The dataframe containing the data for the dashboard.

    Methods:
        generate_dashboard: Generates the dashboard plots.
        save_dashboard: Saves the generated dashboard to a specified directory path.
    """
    def __init__(self, df):

        """
        Initializes the Dashboard instance.

        Args:
            dataframe (pandas.DataFrame): The dataframe containing the data for the dashboard.
        """
        self.df = df

    def generate_dashboard(self):

        """
        Generates individual plots for each metric in the dataframe and combines them into a grid layout.

        Returns:
            bokeh.layouts.gridplot: The grid layout of plots representing the dashboard.

        Raises:
            IndexError: If there are not enough metrics available for plotting.
        """
 
        metrics = list(self.df.columns) 
        plots = []
        colors = Category10[10]  # Change the number based on the number of metrics

        # Generate individual plots with a given color palette
        for i, metric in enumerate(metrics):
            p = figure(title=metric, x_axis_label='Epoch', y_axis_label=metric, width=800, height=300)
            p.line(x=self.df.index, y=self.df[metric], legend_label=metric, color=colors[i],line_width=4)
            plots.append(p)

        # Create grid layout
        self.fig = gridplot(plots, ncols=2)

        return self.fig


    def save_dashboard(self, directory_path):

            """
            Saves the generated dashboard to the specified directory path.

            Args:
                fig (bokeh.layouts.gridplot): The grid layout of plots representing the dashboard.
                directory_path (str): The path to the directory where the dashboard should be saved.
            """

            filename = os.path.join(directory_path,'validation_metrics_log.html')
            output_file(filename=filename, title='validation metrics log')
            save(self.fig, filename)