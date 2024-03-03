import copy
import datetime
import json
import os
import random
import warnings

import numpy as np
import pandas as pd
import timm
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import yaml
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.nn import functional as F
from torch.utils.data import ConcatDataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from datasets import EvalDataset, TrainDataset
from eval import auto_eval
from utils import *

warnings.simplefilter('ignore')


def seed_everything(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(config):
    """Main function for training and evaluating the model.

    Args:
        config (dict): Dictionary of configurations.
    """
    #load conf file for training
    PREDICTION_DIR = config['prediction_dir']
    MODEL_ARCHITECTURE = config['model_architecture']
    PRETRAINED = config['pretrained']
    RESIZE = config['resize']
    IMAGE_NET_NORMALIZE = config['image_net_normalize']
    PREPROCESSING = config['preprocessing']
    NUMBER_KFOLD = config['number_kfold']
    DATA_AUGMENTATION = config['data_augmentation']
    SEED = config['seed']
    LR = float(config['lr'])
    BATCH_SIZE = config['batch_size']
    NUM_EPOCHS = config['epochs']
    GPU_DEVICE = config['gpu_device']
    LOSS_FUNC = config['loss']
    AUTO_EVAL = config['auto_eval']
    # Seed for reproductibility training
    seed_everything(seed=SEED)
    start_training_date = datetime.datetime.now()
    logger.info("start training session '{}'".format(start_training_date))
    date = start_training_date.strftime('%Y_%m_%d_%H_%M_%S')

    if not os.path.exists(PREDICTION_DIR):
        os.makedirs(PREDICTION_DIR)

    prediction_dir = os.path.join(PREDICTION_DIR, '{}'.format(date))
    os.makedirs(prediction_dir)
    log_filename = os.path.join(prediction_dir, "train.log")
    logger.add(log_filename, backtrace=False, diagnose=True)

    
    folds_val_f1 = []
    dataset_path = pd.read_csv('data_splits/train_path.csv')
    stratified_kfold = StratifiedKFold(n_splits=NUMBER_KFOLD)

    for fold, (train_indices, val_indices) in enumerate(stratified_kfold.split(dataset_path.image_path, dataset_path.target)):

        models_path = []
        best_epoch = 0
        best_loss = 0.0
        step = 0
        metrics_dict = {}
        logger.info(f"Fold {fold}:")
        model = timm.create_model(MODEL_ARCHITECTURE, pretrained=PRETRAINED, num_classes=1)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        logger.info("Number of GPU(s) {}: ".format(torch.cuda.device_count()))
        logger.info("GPU(s) in used {}: ".format(GPU_DEVICE))
        logger.info("------")
        device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        model = model.to(device='cuda')
        nb_parameters = count_model_parameters(model=model)
        logger.info("Number of parameters {}: ".format(nb_parameters))
        model.train()

        fold_df_train = dataset_path.iloc[train_indices]
        fold_df_val = dataset_path.iloc[val_indices]
        
        train_dataset = TrainDataset(df_path=fold_df_train, normalize=IMAGE_NET_NORMALIZE, resize=RESIZE, preprocessing=PREPROCESSING, data_augmentation=DATA_AUGMENTATION)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        eval_dataset = EvalDataset(df_path=fold_df_val, preprocessing=PREPROCESSING, resize=RESIZE, normalize=IMAGE_NET_NORMALIZE)
        eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1, shuffle=False)

        fold_val_f1 = []
        model_name = f"{fold}_model.pth"
        model_path = os.path.join(prediction_dir, model_name)

        for epoch in range(NUM_EPOCHS):

            train_losses = AverageMeter()
            eval_losses = AverageMeter()
        
            with tqdm(total=(len(train_dataset) - len(train_dataset) % BATCH_SIZE), ncols = 100, colour='#3eedc4') as t:
                t.set_description('epoch: {}/{}'.format(epoch, NUM_EPOCHS - 1))
                for data in train_dataloader:
                    optimizer.zero_grad()
                    images_inputs, targets = data
                    images_inputs = images_inputs.to(device)
                    targets = targets.to(device)
                    preds = model(images_inputs)
                    preds = torch.sigmoid(preds)
                    targets = torch.unsqueeze(targets, 1)

                    if LOSS_FUNC=="BCE":
                        criterion = nn.BCELoss()
                        loss_train = criterion(preds.to(torch.float32), targets.to(torch.float32))

                    loss_train.backward()
                    optimizer.step()
                    train_losses.update(loss_train.item(), len(images_inputs))
                    t.set_postfix(loss='{:.6f}'.format(train_losses.avg))
                    t.update(len(images_inputs))
                    step += 1

            model.eval()
            targets = []
            preds = []

            # Model Evaluation
            for index, data in enumerate(eval_dataloader):
                images_inputs, target = data
                images_inputs = images_inputs.to(device)
                target = target.to(device)
                target = torch.unsqueeze(target, 1)

                with torch.no_grad():

                    pred = model(images_inputs)
                    pred = torch.sigmoid(pred)
                    if LOSS_FUNC=="BCE":
                        criterion = nn.BCELoss()
                        eval_loss = criterion(pred.to(torch.float32), target.to(torch.float32))

                eval_losses.update(eval_loss.item(), len(images_inputs))
                target = torch.squeeze(target,0)
                target = target.detach().cpu().numpy()
                pred = torch.squeeze(pred,0)
                pred = pred.detach().cpu().numpy()
                pred = pred.round()
                targets.append(target)
                preds.append(pred)
            
            acc = accuracy_score(preds, targets)
            f1 = f1_score(preds, targets)        
            metrics_dict[epoch] = { "F1": f1, "Accuracy": acc, 
                                    'loss_train': train_losses.avg,
                                    'loss_eval': eval_losses.avg}

            fold_val_f1.append(f1)
            loss_plot_filename  = os.path.join(prediction_dir,f"loss_fold_{fold}.png")
            plot_loss_metrics(metrics=metrics_dict, save_path=loss_plot_filename)
            logger.info(f'Epoch {epoch} Eval {LOSS_FUNC} - Loss: {eval_losses.avg} - Acc {acc} - F1 {f1}')
            
            if epoch == 0:

                best_epoch = epoch
                best_f1 = f1
                best_loss = eval_losses.avg
                best_weights = copy.deepcopy(model.state_dict())

            elif f1 > best_f1:

                best_epoch = epoch
                best_f1 = f1
                best_loss = eval_losses.avg
                best_weights = copy.deepcopy(model.state_dict())
            
            torch.save(best_weights, model_path)

        models_path.append(model_path)
        logger.info(f'FOLD {fold} - AVG F1: {np.mean(fold_val_f1)}')
        folds_val_f1.append(np.mean(fold_val_f1))
        logger.info(f'best epoch: {best_epoch}, best F1-score: {best_f1} loss: {best_loss}')

    avg_f1_score = np.mean(folds_val_f1)
    logger.info(f'AVG F1 score: {avg_f1_score}')
    
    config_filename = os.path.join(prediction_dir,'training_config.json')
    with open(config_filename, 'w') as fp:
            json.dump(config, fp)
    # Measure total training time
    end_training_date = datetime.datetime.now()
    training_duration = end_training_date - start_training_date
    logger.info('Training Duration: {}'.format(str(training_duration)))
    model_size = estimate_model_size(model)
    logger.info("model size: {}".format(model_size))
    # Evaluation Ensemble Model
    logger.info("##############")
    logger.info("EVALUATION")
    logger.info("##############")

    list_dir = os.listdir(prediction_dir)
    models_path = [file for file in list_dir if file.endswith('.pth')]
    
    if AUTO_EVAL:

        df_val = pd.DataFrame()
        for i in range(len(models_path)):

            model_path = os.path.join(prediction_dir, models_path[i])
            logger.info(f"model_{i}: {models_path[i]}")
            preds_eval, targets_eval = auto_eval(model_path=model_path,
                                                 model_architecture=MODEL_ARCHITECTURE,
                                                 preprocessing=PREPROCESSING,
                                                 resize=RESIZE,
                                                 normalize=IMAGE_NET_NORMALIZE,
                                                 save_path=prediction_dir)
            model_name = f"model_{i}"
            df_val[model_name] = preds_eval

        df_val['majority'] = df_val.mode(axis=1)[0]
        df_val['target'] = targets_eval
        df_val.to_csv(os.path.join(prediction_dir,'ensemble_model.csv'))
        # Confusion matrix of the ensemble model
        ensemble_pred = df_val['majority'].values
        ensemble_pred = [arr.astype(int) for arr in ensemble_pred]
        plot_confusion_matrix(ensemble_pred, targets_eval, prediction_dir)
        f1_eval = f1_score(ensemble_pred, targets_eval)
        logger.info(f"ENSEMBLE MODEL F1-SCORE: {f1_eval}")
        eval_metric = {'F1': f1_eval}
        save_path = os.path.join(prediction_dir, 'eval_metrics.json')
        
        with open(save_path, 'w') as fp:
            json.dump(eval_metric, fp)  

if __name__ == '__main__':

    with open("config.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logger.info(exc)

    main(config)
