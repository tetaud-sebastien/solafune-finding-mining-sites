import os
import torch

import pandas as pd
from loguru import logger
from datasets import EvalDataset
from torch.utils.data.dataloader import DataLoader
import timm

from sklearn.metrics import accuracy_score, f1_score

import warnings
warnings.filterwarnings("ignore")

def auto_eval(model_path, model_architecture, preprocessing,resize, normalize, save_path):
    """
    Main function to test the trained model on the given test data.

    Args:
        config (dict): The configuration dictionary for the test.
        args (argparse.Namespace): The command-line arguments.

    """
    # import torch.nn as nn
    # from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
    # weights = EfficientNet_V2_L_Weights.IMAGENET1K_V1
    # model = efficientnet_v2_m(weights=None)
    # if 'classifier' in dir(model):
    #     model.classifier[1] = nn.Linear(1280, 1)
    # elif 'fc' in dir(model):
    #     model.fc = nn.Linear(model.fc.in_features, 1)

    model = timm.create_model(model_architecture, pretrained=False, num_classes=1)
    logger.info("==> Loading checkpoint '{}'".format(model_path))
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model.eval()
    model.cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = model.to(device=device)
    logger.info(f"Model is on Cuda: {next(model.parameters()).is_cuda}")
    
    test_path = pd.read_csv('data_splits/test_path.csv')
    test_dataset = EvalDataset(df_path=test_path, preprocessing=preprocessing, resize=resize, normalize=normalize)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    targets_eval = []
    preds_eval = []
    for index, data in enumerate(test_dataloader):
        images_inputs, target = data
        images_inputs = images_inputs.to(device)
        target = target.to(device)
        target = torch.unsqueeze(target, 1)
        with torch.no_grad():

            pred = model(images_inputs)
            pred = torch.sigmoid(pred)

        target = torch.squeeze(target,0)
        target = target.detach().cpu().numpy()
        pred = torch.squeeze(pred,0)
        pred = pred.detach().cpu().numpy()
        pred = pred.round()
        targets_eval.append(target)
        preds_eval.append(pred)
    
    acc = accuracy_score(targets_eval, preds_eval)
    f1 = f1_score(targets_eval, preds_eval)

    metrics = {'accuracy': acc, 'F1': f1}
    save_path = os.path.join(save_path, 'eval_metrics.json')
    import json
    with open(save_path, 'w') as fp:
        json.dump(metrics, fp)

    logger.info(f'EVALUATION: Accuracy {acc} - F1-score: {f1}')
    logger.info("Done")
    return preds_eval, targets_eval

# if __name__ == '__main__':
#     auto_eval(model_path="/home/sebastien/Documents/projects/solafune-finding-mining-sites/training_prediction/2024_02_29_15_11_33/4_model.pth",
#               model_architecture="caformer_s18.sail_in1k",
#               preprocessing="RGB",
#               resize=[512,512],
#               normalize=False, 
#               save_path="/home/sebastien/Documents/projects/solafune-finding-mining-sites/training_prediction")
