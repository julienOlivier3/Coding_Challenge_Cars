import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from torch import nn
import torch.nn.functional as F
warnings.simplefilter(action='ignore', category=FutureWarning)


def plot_describe(df, columns, hue=None, bins=50, bw_method=0.1, size=(24, 12), ncols=3, hspace=0.4, wspace=0.25, log_dict=None):
    """Function to visualize columns in df. Visualization type depends on data type of the column.
    
    Arguments
    ---------
    df : pandas.DataFrame
        Dataframe whose columns shall be visualized.
    columns : list
        Subset of columns which shall be considered only.
    hue: str
        Column according to which single visualization shall be grouped.
    bins : int
        Number of bins for the histogram plots.
    bw_method : float
        method for determining the smoothing bandwidth to use.
    size: tuple
        Size of the resulting grid.
    nclos: int
        Number of columns in the resulting grid.
    hspace : float
        Horizontal space between subplots.
    wspace : float
        Vertical space between subplots.
    log_dict : dict
        Dictionary listing whether a column's visualization should be 
        displayed in log scale on the vertical axis
            

    Returns
    -------
    Visualization of each variable in columns as barplot or histogram.

    """

    # Reduce df to relevant columns
    df = df[columns]
    
    # Calculate the number of rows and columns for the grid
    num_cols = len(df.columns)
    num_rows = int(num_cols / ncols) + (num_cols % ncols)
    
    # Create the subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=ncols, figsize=size)

    # Change the vertical and horizontal spacing between subplots
    plt.subplots_adjust(hspace=hspace, wspace=wspace)  
    
    # Flatten the axes array for easier iteration
    axes = axes.flatten()

    # Do not display vertical axis in log scale as default
    logy = False
    
    # Iterate over each column and plot accordingly
    for i, column in enumerate(df.columns):
        if log_dict!=None:
            logy=log_dict.get(column, False)
            
        ax = axes[i]
        # Barplots for categorical features or integers with few distinct values
        if (df[column].dtype == 'int64' and df[column].value_counts().shape[0] < 10) or df[column].dtype == 'object' or df[column].dtype == '<M8[ns]':
            if hue==None or hue==column:
                temp = df[column].value_counts().sort_index()
                if temp.shape[0] > 20:
                    fontsize = 'xx-small'
                else:
                    fontsize = 'medium'
                temp.plot(kind='bar', ax=ax, ylabel='Count', xlabel='', title=column, logy=logy, fontsize=fontsize)
            else:
                temp = df[[column, hue]].groupby(hue).value_counts(normalize=True).sort_index().to_frame().reset_index()
                if temp.shape[0] > 20:
                    fontsize = 'xx-small'
                else:
                    fontsize = 'medium'
                temp[hue] = temp[hue].astype(str)
                p = sns.barplot(temp, x=column, y='proportion', hue=hue, errorbar=None, ax=ax)
                # Add title and labels
                p.set_title(column)
                p.set_xlabel('')
                p.set_ylabel('Proportion')
                p.set_xticks(p.get_xticks())
                p.set_xticklabels(
                    p.get_xticklabels(), 
                    rotation=90, 
                    horizontalalignment='center', 
                    fontsize=fontsize)
                if logy:
                    p.set_yscale("log")

        # Histograms for floats or integers with many distinct values
        elif (df[column].dtype == 'int64' and df[column].value_counts().shape[0] >= 10) or df[column].dtype == 'float64':
            if hue==None:
                df[column].plot(kind='hist', ax=ax, bins=bins, title=column, logy=logy)
            else:
                hue_groups = np.sort(df[hue].unique())
                for hue_group in hue_groups:
                    p = sns.kdeplot(data=df[df[hue] == hue_group], x=column, fill=True, label=hue_group, ax=ax, bw_method=bw_method)
                # Add title and labels
                p.set_title(column)
                p.set_xlabel('')
                p.set_ylabel('Density')
                p.legend(title=hue)
                if logy:
                    p.set_yscale("log")
                                
        # For all other data types pass
        else:
            pass


class ClassifierModule(nn.Module):
    """
    Feed-forward neural network classifier module.

    Arguments
    ---------
        input_size : int
            Number of features fed into NN.
        num_units : int
            Number of units in the hidden layer (default: 10).
        nonlin : function
            Activation function for the hidden layer (default: F.relu).
        dropout : float
            Dropout probability (default: 0.5).
        output_size : int
            Size of the output layer (default: 3).
    """

    def __init__(self, input_size=97, num_units=10, nonlin=F.relu, dropout=0.5, output_size=3):        
        super(ClassifierModule, self).__init__()
        self.num_units = num_units
        self.nonlin = nonlin
        self.dropout = dropout
        self.input_size = input_size
        self.output_size = output_size
        

        self.dense0 = nn.Linear(input_size, num_units)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(dropout)
        self.dense1 = nn.Linear(num_units, 10)
        self.output = nn.Linear(10, output_size)

    def forward(self, X):
        """
        Forward pass of the neural network.

        Arguments
        ---------
            X : torch.Tensor
                Input tensor.

        Returns
        -------
            y : torch.Tensor
                Output tensor.
        """
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = F.relu(self.dense1(X))
        y = F.softmax(self.output(X), dim=-1)
        return y
    


class RegressorModule(nn.Module):
    """
    Feed-forward neural network regression module.

    Arguments
    ---------
        input_size : int
            Number of features fed into NN.
        num_units : int
            Number of units in the hidden layer (default: 10).
        nonlin : function
            Activation function for the hidden layer (default: F.relu).
    """
    def __init__(
            self,
            input_size=97, 
            num_units=10,
            nonlin=F.relu,
    ):
        super(RegressorModule, self).__init__()
        self.input_size = input_size
        self.num_units = num_units
        self.nonlin = nonlin

        self.dense0 = nn.Linear(input_size, num_units)
        self.nonlin = nonlin
        self.dense1 = nn.Linear(num_units, 10)
        self.output = nn.Linear(10, 1)

    def forward(self, X):
        """
        Forward pass of the neural network.

        Arguments
        ---------
            X : torch.Tensor
                Input tensor.

        Returns
        -------
            y : torch.Tensor
                Output tensor.
        """
        X = self.nonlin(self.dense0(X))
        X = F.relu(self.dense1(X))
        y = self.output(X)
        y = y.flatten(start_dim=1)
        y = y.squeeze()
        return y