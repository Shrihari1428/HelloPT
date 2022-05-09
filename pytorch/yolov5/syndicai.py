import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

class BinaryClassification(nn.Module):
    def __init__(self):
        super(BinaryClassification, self).__init__()
        # Number of input features is 12.
        self.layer_1 = nn.Linear(330, 256) 
        self.layer_2 = nn.Linear(256, 256)
        self.layer_out = nn.Linear(256, 1) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(256)
        self.batchnorm2 = nn.BatchNorm1d(256)
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        
        return x

class PythonPredictor:

    def __init__(self, config):
        """ Download pretrained model. """
        self.model = BinaryClassification()
        self.model = torch.load('Model.pt').autoshape()

    def predict(self, payload):
        """ Run a model based on url input. """

        # Inference
        y = self.model(payload["sample"])

        # # Draw boxes
        # boxes = results.xyxy[0].numpy()
        # box_img = draw_box(img, boxes)

        # # Save image
        # #box_img.save("sample_data/output.png", "PNG")

        return y