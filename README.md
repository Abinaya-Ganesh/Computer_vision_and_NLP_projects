# Computer_vision_and_NLP_projects
Various CCN models' performance is evaluated and compared for MNIST, Fashion MNIST and CIFAR-10 Datasets | Importance of attention in Seq2seq models | Creating a front end for Multifunctional NLP and Image generation models | Python | Machine Learning | Deep Learning | PyTorch | Computer Vision | NLP | Streamlit

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

  **a.For Deep Learning**

    • torch

    • torchvision

    • torch-utils

 **3.Python modules to import**

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

  **e.For visualization**

    • import matplotlib.pyplot as plt

 **Process**

    • Load and preprocess the datasets MNIST, FMNIST, CIFAR-10.

    • Implement the following CNN architectures: LeNet-5, AlexNet, GoogLeNet, VGGNet, ResNet18 and SENet18.
    
    • Train each model on each dataset, recording the loss and accuracy metrics.

    • Evaluate the performance of each model on the test sets using accuracy, precision, recall, and F1-score.

    • Plot the loss curves for comparison.

    • Tabulate other performance metrics for comparison.
    
    • Analyze the results to understand the impact of different architectures' performance on different datasets.


  **NOTE:**

  1. CNN_project file has all the architectures, training and evaluation part.
     
  2. The loss and other metrics are pickled. I have uploaded the pickled files as well.
     
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

  **a.For Deep Learning**

    • torch

    • torch-utils

 **3.Python modules to import**

  **a.For Neural Network models**

    • import torch
    
    • import torch.nn as nn

    • import torch.optim as optim

  **b.For Datasets**
    
    • from torch.utils.data import Dataset, DataLoader

  **c.For evaluation metrics**

    • from sklearn.metrics import accuracy_score

  **d.For file handling**

    • import pandas as pd

  **e.For visualization**

    • import matplotlib.pyplot as plt

**Process**

    • Generate a synthetic dataset where each source sequence is a random sequence of integers, and each target sequence is the reverse of the source sequence. 
    
    • This data is used to understand the model's behavior with ease.

    • Implement the sequence-to-sequence model **without attention** mechanism in PyTorch.
    
    • Implement the sequence-to-sequence model **with attention** mechanism in PyTorch.

    • Train the models on the synthetic dataset.

    • Get the loss curves for the seq2seq model with and without attention mechanism during training.

    • Plot the loss curves for comparison.

    • Evaluate the model performance using metrics such as accuracy.
    
    • Analyze the effectiveness of the attention mechanism in improving seq2seq model performance.

**NOTE:**

    The seq2seq_project ipynb file consists of model architectures, synthetic data generation, training and evaluation and plot for loss.

**PROJECT-3**

**Multifunctional NLP and Image Generation Tool using Hugging Face Models**

The goal of this project is to create a multifunctional tool that allows users to select and utilize different pretrained models from Hugging Face for various tasks. The tool will support text summarization, next word prediction, story prediction, chatbot, sentiment analysis, question answering, and image generation. The front end will provide a user-friendly interface to select the task and input the required text or image for processing.

I have worked with Hugging Face models using PyTorch Library here and the models run on **GPU**. Streamlit is run on Google Colab by creating a tunnel to render the web application.

**Developer Guide**

**1.Tools required**

  • Python

  • Google Colab

**2.Python libraries to install**
  **a.For dashboard creation**

    • Streamlit

    • npx

  **b.For Deep Learning**

    • torch

    • transformers

    • diffusers


 **3.Python modules to import**

  **a.For Hugging face models**

    • import torch

    • from transformers import GPT2LMHeadModel, GPT2Tokenizer

    • from transformers import AutoModelForCausalLM, AutoTokenizer

    • from transformers import pipeline

    • from transformers import BartForConditionalGeneration, BartTokenizer

    • from transformers import AutoTokenizer, AutoModelForQuestionAnswering

    • from diffusers import StableDiffusionPipeline

  **b.Dashboard Libraries**

    • import streamlit as st

  **Process**

    • Set up the environment and install necessary libraries, including Hugging Face Transformers.

    • Implement a user-friendly front end for task selection and input. I have used Streamlit here.

    • Load and integrate pretrained models from Hugging Face for the following tasks:
       - Text Summarization
       - Next Word Prediction
       - Story Prediction
       - Chatbot
       - Sentiment Analysis
       - Question Answering
       - Image Generation

    • Implement the backend logic to process user inputs and generate outputs using the selected models.

    • Test the application with various inputs and refine the user interface and backend logic.

 **Implementing Streamlit on Google colab**

    • Since this project requires GPU, I have used Google Colab and implemented Streamlit via colab.
    
    • Using the command %%writefile app.py, write all the necessary codes for Streamlit in a single cell in Google Colab.
    
    • Run this command !wget -q -O - ipv4.icanhazip.com to get a password for local tunnel.

  ![screenshot](https://github.com/Abinaya-Ganesh/Computer_vision_and_NLP_projects/assets/162968618/30114128-9cfa-4347-8307-e53045f26af1)

    • Using this command !streamlit run app.py & npx localtunnel --port 8501 create a local tunnel and run in the Streamlit app on browser

  ![image](https://github.com/Abinaya-Ganesh/Computer_vision_and_NLP_projects/assets/162968618/29471ac7-8d56-4afc-b674-33c7a255b92c)

    • Your Streamlit app is here!

  ![image](https://github.com/Abinaya-Ganesh/Computer_vision_and_NLP_projects/assets/162968618/041aa011-75e6-4464-9ddd-d3a20456c2bb)
