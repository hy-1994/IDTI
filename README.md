# IDTI: An increment of diversity method for cell trajectory inference of time-series scRNA-seq data
### Overview：
![image-20220808170723505](workflow.png)

We present IDTI based on the increment of diversity for trajectory inference from time-series scRNA-Seq data, which combines time series information and increment of diversity method to infer cell state trajectory.


### Systems Requirements

The scripts were written in Python language.

To run the scripts, you need several python packages, as follow:

import os  
import numpy as np  
import pandas as pd  
import scanpy as sc   
import anndata as ad  
import math  
import operator  
import collections  
import networkx as nx  
import matplotlib.pyplot as plt  
from collections import Counter  
from sklearn.preprocessing import MinMaxScaler  
from networkx.drawing.nx_agraph import graphviz_layout

### Usage

There are two main folders:

The folder "test_data" contains test data, which is the simulated time-series scRNA-seq data.
1. Subfile "sim_counts.csv" is the gene expression matrix of simulatede data.
2. Subfile "sim_metadata" is the metadata of simulatede data.
3. Subfile "sim_data.h5ad" is the H5AD file of simulatede data.

The folder "script" contains main scripts.
"core.py" is uesed to trajectory inference of IDTI.

### References

IDTI: An increment of diversity method for cell trajectory inference of time-series scRNA-seq data

Contact: 31709150@mail.imu.edu.cn

