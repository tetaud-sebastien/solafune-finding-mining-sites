import os 
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import confusion_matrix
from torchvision.transforms import ToTensor
from utils import * 

from loguru import logger

MAX_TENSORBOARD_IMAGES = 5  # Adjust as needed


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def train_log(step, loss, tensorboard_writer, name, **kwargs):


    
    if step % 100 == 0:
        loss = loss.detach().cpu().numpy()
        if tensorboard_writer is not None:
            tensorboard_writer.add_scalar(name+'/loss', loss, step)


def val_log(epoch, step, images_inputs, seg_targets, seg_preds,
            tensorboard_writer, name, prediction_dir, **kwargs):
    """
    Logs validation-related information and visualizations to TensorBoard.

    Args:
        step (int): Current step or epoch.
        input_batch (torch.Tensor): Input batch of images.
        target (torch.Tensor): True target labels.
        output_batch (torch.Tensor): Model's predicted output batch.
        labels (dict): Dictionary mapping class indices to class labels.
        confusion_matrix (np.ndarray): Confusion matrix for evaluation.
        tensorboard_writer (SummaryWriter): TensorBoard writer for logging.
        name (str): Name prefix for the logs.
        prediction_dir (str): Directory to save prediction visualizations.
        **kwargs: Additional keyword arguments (e.g., loss).

    Returns:
        None 
    """

    if not os.path.exists(prediction_dir):
        print("Create prediction directory...")
        os.makedirs(prediction_dir, exist_ok=True)
    now = datetime.now()
    time = now.strftime("%m_%d_%Y_%H_%M_%S")

    # Extract loss value from kwargs
    loss = kwargs['loss'].detach().cpu().numpy()

    if tensorboard_writer is not None:
        tensorboard_writer.add_scalar(name + '/loss', loss, step)


    seg_targets = seg_targets.detach().cpu().numpy()
    seg_preds = seg_preds.detach().cpu().numpy()


    if step % 100 == 0:
        for i in range(min(images_inputs.shape[0], MAX_TENSORBOARD_IMAGES)):

        
            seg_pred = seg_preds[i]
            mask = seg_targets[i]
            image = images_inputs[i]
            
            #Apply thresholding to convert values to 0 or 1
            binary_predictions = (seg_pred > 0.5).astype(np.uint8)
            binary_predictions = binary_predictions[0,:,:]

            image = torch.permute(image, (2, 1, 0))
            image = image.detach().cpu().numpy()
            # Scale image values to the range [0, 255]
            image = ((image - image.min()) / (image.max() - image.min())) * 255
            image = image.astype(np.uint8)

            mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            mask_rgb[..., 0] = mask * 255
            mask_rgb[..., 1] = mask * 255
            mask_rgb[..., 2] = mask * 255

            pred_rgb = np.zeros((binary_predictions.shape[0], binary_predictions.shape[1], 3), dtype=np.uint8)
            pred_rgb[..., 0] = binary_predictions * 255  
            pred_rgb[..., 1] = binary_predictions * 255  
            pred_rgb[..., 2] = binary_predictions * 255

        
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_tensorboard = np.hstack((image, mask_rgb, pred_rgb))
            # # Add the input image to TensorBoard
            tensorboard_writer.add_image(f'{name}/input', image_tensorboard, dataformats='HWC')
            # # Save the image with prediction label to the prediction directory
            now = datetime.now()
            time = now.strftime("%m_%d_%Y_%H_%M_%S")
            filename = str(epoch)+ "_" + str(step) + '.png'
            filename = os.path.join(prediction_dir, f'{time}_{filename}')
            cv2.imwrite(filename, image_tensorboard)
