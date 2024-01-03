import os 
import pandas as pd
import torch
import seaborn as sns

from bokeh.plotting import figure, show
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource
from bokeh.palettes import Category10
from bokeh.plotting import figure, output_file, save
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_pred, y_true, save_path):
    """
    Plot confusion matrices using Seaborn's heatmap for two predicted sets.

    Args:
    
    - y_pred (array-like): Predicted labels for the first set.
    - y_true (array-like): True labels.
    """

    
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(16, 8))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))


def plot_loss_metrics(metrics, save_path):

    """
    Generate a plot displaying 'loss_train' and 'loss_eval' metrics against epochs.

    Parameters:
    metrics (dict): A dictionary containing metrics for different epochs.
                    Each key represents an epoch number, and the corresponding value is a dictionary
                    containing metrics such as 'loss_train' and 'loss_eval'.

    Returns:
    None: Displays a plot showing 'loss_train' and 'loss_eval' against the epochs.
    """
    epochs = list(metrics.keys())
    loss_train = [metrics[epoch]['loss_train'] for epoch in epochs]
    loss_eval = [metrics[epoch]['loss_eval'] for epoch in epochs]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss_train, label='Loss Train')
    plt.plot(epochs, loss_eval, label='Loss Eval')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Evaluation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'loss_plot.png'))  # Save the plot as an image
    plt.show()

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


class Dashboard:

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