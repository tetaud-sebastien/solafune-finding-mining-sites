import copy
import datetime
import os
import shlex
import subprocess
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T

from tqdm import tqdm

from datasets_solafune import TrainDataset, EvalDataset
from utils import *
from logs.train import train_log, val_log
import segmentation_models_pytorch as smp

from loguru import logger
import segmentation_models_pytorch as smp

import warnings
warnings.simplefilter('ignore')

def main(config):
    """Main function for training and evaluating the model.

    Args:
        config (dict): Dictionary of configurations.
    """

    # load conf file for training
    dataset_name = config['dataset_name']
    output_dir = config['outpout_dir']
    prediction_dir = config['prediction_dir']
    seed = config['seed']
    lr = float(config['lr'])
    batch_size = config['batch_size']
    num_epochs = config['epochs']
    log_dir = config['log_dir']
    tbp = config['tbp']
    gpu_device = config['gpu_device']
    loss_func = config['loss']
    threshold = config['threshold']
    
    start_training_date = datetime.datetime.now()
    logger.info("start training session '{}'".format(start_training_date))
    date = start_training_date.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = os.path.join(output_dir, '{}'.format(date))
    log_filename = os.path.join(output_dir, "train.log")
    logger.add(log_filename, backtrace=False, diagnose=True)
    logger.info("output directory: {}".format(output_dir))
    logger.info("------")

    TENSORBOARD_DIR = 'tensorboard'
    tensorboard_path = os.path.join(log_dir, TENSORBOARD_DIR)
    logger.info("Tensorboard path: {}".format(tensorboard_path))
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path, exist_ok=True)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    prediction_dir = os.path.join(prediction_dir, '{}'.format(date))
    os.makedirs(prediction_dir)

    cudnn.benchmark = True
    if tbp is not None:

        logger.info("starting tensorboard")
        logger.info("------")

        command = f'tensorboard --logdir {tensorboard_path} --port {tbp} --host 10.0.0.6 --load_fast=true'
        tensorboard_process = subprocess.Popen(shlex.split(command), env=os.environ.copy())

        train_tensorboard_writer = SummaryWriter(
            os.path.join(tensorboard_path, 'train'), flush_secs=30)
        val_tensorboard_writer = SummaryWriter(
            os.path.join(tensorboard_path, 'val'), flush_secs=30)
        writer = SummaryWriter()
    else:
        logger.exception("An error occurred: {}", "no tensorboard")
        tensorboard_process = None
        train_tensorboard_writer = None
        val_tensorboard_writer = None

    # Seed for reproductibility training
    torch.manual_seed(seed)
    
    model = smp.Unet(encoder_name="resnet50", encoder_weights=None,in_channels=3, classes=1, activation="sigmoid")

    logger.info("Number of GPU(s) {}: ".format(torch.cuda.device_count()))
    logger.info("GPU(s) in used {}: ".format(gpu_device))
    logger.info("------")

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    if len(str(gpu_device)) > 1:
        model = nn.DataParallel(model)

    model = model.to(device='cuda')
    nb_parameters = count_model_parameters(model=model)
    logger.info("Number of parameters {}: ".format(nb_parameters))
    logger.info("------")
    logger.info("Dataset in use: {}".format(dataset_name))
    logger.info("------")

    # Define Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Load train and test data path
    train_path = pd.read_csv('data_splits/train_path.csv')
    # train_path = train_path[:100]
    valid_path = pd.read_csv('data_splits/test_path.csv')
    # valid_path = valid_path[-50:]
    logger.info("Number of Training data {0:d}".format(len(train_path)))
    logger.info("------")
    logger.info("Number of Validation data {0:d}".format(len(valid_path)))
    logger.info("------")

    train_dataset = TrainDataset(df_path=train_path)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=False, num_workers=0)

    eval_dataset = EvalDataset(df_path=valid_path)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1, shuffle=False)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_loss = 0.0
    step = 0
    metrics_dict = {}

    for epoch in range(num_epochs):

        epoch_losses = AverageMeter()
        eval_losses = AverageMeter()

        model.train()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % batch_size), ncols = 100, colour='#3eedc4') as t:
            t.set_description('epoch: {}/{}'.format(epoch, num_epochs - 1))

            for data in train_dataloader:

                optimizer.zero_grad()
                # inputs, targets = data
                images_inputs, seg_targets = data
                images_inputs = images_inputs.to(device)
                seg_targets = seg_targets.to(device)
                seg_preds = model(images_inputs)

                # Compute loss
                if loss_func == "cross_entropy":

                    criterion = nn.CrossEntropyLoss()
                    loss_train = criterion(seg_preds.to(
                        torch.float32), seg_targets.to(torch.float32))

                if loss_func == "Dice":

                    criterion = DiceLoss()
                    loss_train = criterion(seg_preds, seg_targets)

                elif loss_func == "WeightedBCE":

                    criterion = WeightedBCELoss(weight_pos=2, weight_neg=1)
                    loss_train = criterion(seg_preds[:, 0, :, :].to(
                        torch.float32), seg_targets.to(torch.float32))

                elif loss_func == "BCE":

                    criterion_seg = nn.BCELoss()
                    loss_train = criterion_seg(seg_preds[:, 0, :, :].to(torch.float32), seg_targets.to(torch.float32))

                loss = loss_train

                loss.backward()
                optimizer.step()

                epoch_losses.update(loss.item(), len(images_inputs))
                train_log(step=step, loss=loss, tensorboard_writer=train_tensorboard_writer, name="Training")

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(images_inputs))

                step += 1

        # Save model
        torch.save(model.state_dict(), os.path.join(output_dir, 'epoch_{}.ckpt'.format(epoch)))
        model.eval()

        iou_metrics = []
        f1_score_metrics = []
        f2_score_metrics = []
        accuracy_metrics = []
        recall_metrics = []
        metrics_dict = {}

        # Model Evaluation
        for index, data in enumerate(eval_dataloader):
            
        

            images_inputs, seg_targets = data
            images_inputs = images_inputs.to(device)
            seg_targets = seg_targets.to(device)

            with torch.no_grad():

                seg_preds = model(images_inputs)

            if loss_func == "cross_entropy":

                criterion = nn.CrossEntropyLoss()
                eval_loss = criterion(seg_preds.to(
                    torch.float32), seg_targets.to(torch.float32))

            if loss_func == "Dice":
                eval_loss = criterion(seg_preds, seg_targets)

            elif loss_func == "WeightedBCE":

                criterion = WeightedBCELoss(weight_pos=2, weight_neg=1)

                eval_loss = criterion(seg_preds[:, 0, :, :].to(
                    torch.float32), seg_targets.to(torch.float32))

            elif loss_func == "BCE":

                criterion = BCELoss()
                eval_loss = criterion_seg(seg_preds[:, 0, :, :].to(
                    torch.float32), seg_targets.to(torch.float32))

            # val_log(epoch=epoch, step=index, loss=eval_loss, images_inputs=images_inputs,
            #         seg_targets=seg_targets, seg_preds=seg_preds,
            #         tensorboard_writer=val_tensorboard_writer, name="Validation",
            #         prediction_dir=prediction_dir)

            eval_losses.update(eval_loss.item(), len(images_inputs))
            preds = seg_preds.detach().cpu().numpy()
            targets = seg_targets.detach().cpu().numpy()

            threshold = 0.5
            binary_prediction = (preds[0,0,:,:] > threshold).astype(np.uint8)
            binary_prediction = binary_prediction.flatten()
            seg_target = targets[0,:,:].flatten()


            from sklearn.metrics import f1_score
            from sklearn.metrics import jaccard_score

            f1_score_metrics.append(f1_score(seg_target, binary_prediction))
            iou_metrics.append(jaccard_score(seg_target, binary_prediction))
        
        metrics_dict[epoch] = { "IoU": np.mean(iou_metrics),"F1": np.mean(f1_score_metrics)
                               }
        print(metrics_dict)
        df_metrics = pd.DataFrame(metrics_dict).T
        df_mean_metrics = df_metrics.mean()
        df_mean_metrics = pd.DataFrame(df_mean_metrics).T

        if epoch == 0:

            df_val_metrics = pd.DataFrame(columns=df_mean_metrics.columns)
            df_val_metrics = pd.concat([df_val_metrics, df_mean_metrics])

        else:
            df_val_metrics = pd.concat([df_val_metrics, df_mean_metrics])
            df_val_metrics = df_val_metrics.reset_index(drop=True)

        dashboard = Dashboard(df_val_metrics)
        dashboard.generate_dashboard()
        dashboard.save_dashboard(directory_path=prediction_dir)
        # Access the mean values
        logger.info('Epoch {} Eval {} Loss: {:.2f}'.format(
            epoch, loss_func, eval_losses.avg))
        t.write('eval {} Loss: {:.2f}'.format(loss_func, eval_losses.avg))

        # Save best model
        if epoch == 1:

            best_epoch = epoch
            best_loss = eval_losses.avg
            best_weights = copy.deepcopy(model.state_dict())

        elif eval_losses.avg < best_loss:

            best_epoch = epoch
            best_loss = eval_losses.avg
            best_weights = copy.deepcopy(model.state_dict())

    logger.info('best epoch: {}, {} loss: {:.2f}'.format(
        best_epoch, loss_func, best_loss))
    torch.save(best_weights, os.path.join(output_dir, 'best.pth'))
    logger.info('Training Done')
    logger.info('best epoch: {}, {} loss: {:.2f}'.format(
        best_epoch, loss_func, best_loss))
    # Measure total training time
    end_training_date = datetime.datetime.now()
    training_duration = end_training_date - start_training_date
    logger.info('Training Duration: {}'.format(str(training_duration)))
    df_val_metrics['Training_duration'] = training_duration
    df_val_metrics['nb_parameters'] = nb_parameters
    model_size = estimate_model_size(model)
    logger.info("model size: {}".format(model_size))
    df_val_metrics['model_size'] = model_size
    # Save validation metrics
    df_val_metrics.to_csv(os.path.join(
        prediction_dir, 'valid_metrics_log.csv'))


if __name__ == '__main__':

    with open("config.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logger.info(exc)

    main(config)
