import argparse
import torch

import pandas as pd
from loguru import logger
from datasets import EvalDataset
from torch.utils.data.dataloader import DataLoader

from sklearn.metrics import accuracy_score, f1_score
from models import Unet

import warnings
warnings.filterwarnings("ignore")


def auto_eval(model_path, model_architecture,channel):
    """
    Main function to test the trained model on the given test data.

    Args:
        config (dict): The configuration dictionary for the test.
        args (argparse.Namespace): The command-line arguments.

    """
    
    
    # model = Unet(num_channels=channel)
    import timm 
    #model = timm.create_model('tf_efficientnetv2_s.in21k_ft_in1k', num_classes=1)
    model = timm.create_model(model_architecture, pretrained=True, num_classes=1)
    logger.info("==> Loading checkpoint '{}'".format(model_path))
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model.eval()
    model.cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = model.to(device=device)
    logger.info(f"Model is on Cuda: {next(model.parameters()).is_cuda}")
    
    test_path = pd.read_csv('data_splits/test_path.csv')
    test_dataset = EvalDataset(df_path=test_path)
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
    
    acc = accuracy_score(preds_eval, targets_eval)
    f1 = f1_score(preds_eval, targets_eval)

    logger.info(f'EVALUATION: Accuracy {acc} - F1-score: {f1}')
    logger.info("Done")

    return preds_eval, targets_eval

# if __name__ == '__main__':

#     parser = argparse.ArgumentParser(description='Evaluation data generation', fromfile_prefix_chars='@')
#     parser.add_argument('-c', '--checkpoint_path',type=str,   help='path to a checkpoint to load', default='')
#     args = parser.parse_args()
    
#     main(args)
