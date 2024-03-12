import argparse
import os
import warnings

import pandas as pd
import timm
import torch
from loguru import logger
from torch.utils.data.dataloader import DataLoader

from datasets import TestDataset
from models import Unet

warnings.filterwarnings("ignore")


def main(args):
    """
    Main function to test the trained model on the given test data.

    Args:
        config (dict): The configuration dictionary for the test.
        args (argparse.Namespace): The command-line arguments.

    """
    from torch import nn
    import torchvision.models as models

    checkpoint_path = args.checkpoint_path
    list_dir = os.listdir(checkpoint_path)
    resize = [512,512]
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    models_path = [file for file in list_dir if file.endswith('.pth')]
    df_pred = pd.DataFrame()
    for i in range(len(models_path)):
        model_path = os.path.join(checkpoint_path, models_path[i])
        logger.info(f"model: {models_path[i]}")
        model = timm.create_model("davit_base.msft_in1k", pretrained=False, num_classes=1)
        
        # from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights

        # # weights = EfficientNet_V2_L_Weights.IMAGENET1K_V1
        # model = efficientnet_v2_m(weights=None)
        # # model.fc = nn.Linear(model.fc.in_features, 1)

        # # Modify the classifier for your specific classification task
        # if 'classifier' in dir(model):
        #     model.classifier[1] = nn.Linear(1280, 1)
        # elif 'fc' in dir(model):
        #     model.fc = nn.Linear(model.fc.in_features, 1)
        logger.info("==> Loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint)
        # model.eval()
        model.cuda()
        device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        model = model.to(device=device)
        logger.info(f"Model is on Cuda: {next(model.parameters()).is_cuda}")
        
        dfs = pd.read_csv("/home/sebastien/Documents/projects/solafune-finding-mining-sites/data/uploadsample.csv", header=None)
        submit_path = pd.read_csv("/home/sebastien/Documents/projects/solafune-finding-mining-sites/data_splits/submit_path.csv")
        test_dataset = TestDataset(df_path=submit_path, preprocessing= "RGB", resize=resize, normalize=False)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
        preds_submit = []
        for index, data in enumerate(test_dataloader):
        
            images_inputs = data
            images_inputs = images_inputs.to(device)
        
            with torch.no_grad():

                pred = model(images_inputs)
                pred = torch.sigmoid(pred)

            pred = torch.squeeze(pred,0)
            pred = pred.detach().cpu().numpy()
            pred = pred[0].round().astype(int)

            preds_submit.append(pred)

        df_pred[f"model_{i}"] = preds_submit

    df_pred['majority'] = df_pred.mode(axis=1)[0]
    dfs[1] = df_pred['majority']
    dfs[1] = preds_submit
    # some data are not valid
    dfs = dfs.replace(-9223372036854775808, int(0))
    dfs.to_csv("submit.csv", header=False, index=False)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluation data generation', fromfile_prefix_chars='@')
    parser.add_argument('-c', '--checkpoint_path',type=str,   help='path to a checkpoint to load', default='')
    args = parser.parse_args()
    
    main(args)
