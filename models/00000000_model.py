import torch
import torch.nn as nn
from typing import Dict

from . import BaseModel, register_model

@register_model("00000000_model")
class MyModel00000000(BaseModel):

    def __init__(self, input_size=128, hidden_size=512, num_layers=1, dropout_rate=0.2, **kwargs):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define the lstm layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Define the batch normalization layer
        self.batchnorm = nn.BatchNorm1d(hidden_size)
        
        # Define the dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # Define the ReLU activation function
        self.relu = nn.ReLU()

        self.task_layers = nn.ModuleDict({
            "short_mortality": nn.Linear(hidden_size, 1),
            "long_mortality": nn.Linear(hidden_size, 1),
            "readmission": nn.Linear(hidden_size, 1),
            "diagnosis_1": nn.Linear(hidden_size, 1), # 17 diagnoses, each binary
            "diagnosis_2": nn.Linear(hidden_size, 1),
            "diagnosis_3": nn.Linear(hidden_size, 1),
            "diagnosis_4": nn.Linear(hidden_size, 1),
            "diagnosis_5": nn.Linear(hidden_size, 1),
            "diagnosis_6": nn.Linear(hidden_size, 1),
            "diagnosis_7": nn.Linear(hidden_size, 1),
            "diagnosis_8": nn.Linear(hidden_size, 1),
            "diagnosis_9": nn.Linear(hidden_size, 1),
            "diagnosis_10": nn.Linear(hidden_size, 1),
            "diagnosis_11": nn.Linear(hidden_size, 1),
            "diagnosis_12": nn.Linear(hidden_size, 1),
            "diagnosis_13": nn.Linear(hidden_size, 1),
            "diagnosis_14": nn.Linear(hidden_size, 1),
            "diagnosis_15": nn.Linear(hidden_size, 1),
            "diagnosis_16": nn.Linear(hidden_size, 1),
            "diagnosis_17": nn.Linear(hidden_size, 1),
            "short_los": nn.Linear(hidden_size, 1),
            "long_los": nn.Linear(hidden_size, 1),
            "final_acuity": nn.Linear(hidden_size, 6),
            "imminent_discharge": nn.Linear(hidden_size, 6),
            "creatinine_level": nn.Linear(hidden_size, 5),
            "bilirubin_level": nn.Linear(hidden_size, 5),
            "platelet_level": nn.Linear(hidden_size, 5),
            "wbc_level": nn.Linear(hidden_size, 3),
        })
        
    def get_logits(cls, net_output):
        logits = torch.cat([layer for task, layer in net_output.items()], dim=1)
        return logits
    
    def get_targets(self, sample):
        return sample['label']

    def forward(self, data, **kwargs):

        x = data.view(-1, 100, 128)

        # Initialize hidden and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Pass the input through the LSTM layer
        x, _ = self.lstm(x, (h0, c0))
        
        # Apply batch normalization
        x = self.batchnorm(x[:, -1, :])

        # Apply ReLU activation function
        x = self.relu(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Pass the output through the task specific layers
        output = {}
        
        for task, layer in self.task_layers.items():
            output[task] = layer(x)

        return output