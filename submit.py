import argparse
import torch

import pandas as pd
from loguru import logger
from datasets import TestDataset
from torch.utils.data.dataloader import DataLoader

from models import Unet
import warnings
warnings.filterwarnings("ignore")


def main(args):
    """
    Main function to test the trained model on the given test data.

    Args:
        config (dict): The configuration dictionary for the test.
        args (argparse.Namespace): The command-line arguments.

    """
    
    model_path = args.checkpoint_path    
    import timm 
    # model = timm.create_model('tf_efficientnetv2_s.in21k_ft_in1k', num_classes=1)
    model = timm.create_model('caformer_s18.sail_in1k', num_classes=1)
    logger.info("==> Loading checkpoint '{}'".format(model_path))
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model.eval()
    model.cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = model.to(device=device)
    logger.info(f"Model is on Cuda: {next(model.parameters()).is_cuda}")
    
    dfs = pd.read_csv("/home/sebastien/Documents/projects/solafune-finding-mining-sites/data/uploadsample.csv", header=None)
    submit_path = pd.read_csv("/home/sebastien/Documents/projects/solafune-finding-mining-sites/data_splits/submit_path.csv")
    test_dataset = TestDataset(df_path=submit_path)
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

    dfs[1] = preds_submit
    dfs.to_csv("submit.csv", header=False, index=False)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluation data generation', fromfile_prefix_chars='@')
    parser.add_argument('-c', '--checkpoint_path',type=str,   help='path to a checkpoint to load', default='')
    args = parser.parse_args()
    
    main(args)
