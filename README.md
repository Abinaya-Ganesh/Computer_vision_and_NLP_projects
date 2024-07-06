# Computer_vision_and_NLP_projects
Various CCN models' performance is evaluated for MNIST, Fashion MNIST and CIFAR-10 Datasets | Importance of attention in Seq2seq models | Creating a front end for Multifunctional NLP and Image generation models | Python | Machine Learning | Deep Learning | PyTorch | Computer Vision | NLP | Streamlit

This repository consists of three projects.
1. Comparison of CNN Architectures on Different Datasets
2. Sequence-to-Sequence Modeling with Attention Mechanism
3. Multifunctional NLP and Image Generation Tool using Hugging Face Models

**PROJECT-1**

**Comparison of CNN Architectures on Different Datasets**

Various Convolutional Neural Network Architectures like LeNet5, AlexNet, VGG, ResNet18, SENet18 and GoogleNet's performance are evaluated and compared based on different datasets which include MNIST, FashionMNIST and CIFAR-10.

I have worked with PyTorch Library here and the models run on **GPU**.

**Developer Guide**

**1.Tools required**

  • Python

  • Google Colab

 **2.Python libraries to install**

  **a.For Neural Network models**

    • import torch
    
    • import torch.nn as nn

    • import torch.nn.functional as F

  **b.For Datasets**
    
    • from torchvision import datasets

    • import torchvision.transforms as transforms
    
    • from torch.utils.data import DataLoader

 **c.For optimizer**
    
    • import torch.optim as optim

    • from torch.optim.lr_scheduler import StepLR

  **d.For evaluation metrics**
    
    • from sklearn.metrics import precision_score, recall_score, f1_score

  **e.For file handling**

    • import pandas as pd

  **f.Visualization**

    • import matplotlib.pyplot as plt

 **Process**

    • Load and preprocess the datasets MNIST, FMNIST, CIFAR-10.

    • Implement the following CNN architectures: LeNet-5, AlexNet, GoogLeNet, VGGNet, ResNet18 and SENet18.
    
    • Train each model on each dataset, recording the loss and accuracy metrics.

    • Evaluate the performance of each model on the test sets using accuracy, precision, recall, and F1-score.

    • Plot the loss curves for comparison.

    • Tabulate other performance metrics for comparison.
    
    • Analyze the results to understand the impact of different architectures and datasets on model performance.


  **NOTE:**

  1. CNN_project file has all the architectures, training and evaluation part.
     
  2. The loss and other metrics are converted to CSV files.
     
  3. CNN_plot file consists of the plotted loss curve and other evaluation metrics table.

**PROJECT-2**

**Sequence-to-Sequence Modeling with Attention Mechanism**

The goal of this project is to implement and evaluate sequence-to-sequence (seq2seq) models with attention mechanism. We will train the models on a synthetic dataset where the target sequence is the reverse of the source sequence. 

I have worked with PyTorch Library here and the models run on **GPU**.

**Developer Guide**

**1.Tools required**

  • Python

  • Google Colab

 **2.Python libraries to install**

  **a.For Neural Network models**

    • import torch
    
    • import torch.nn as nn

    • import torch
    
    • import torch.nn as nn







