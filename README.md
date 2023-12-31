# Tetherkit Neural Network Utilities
A collection of scripts and snippets to help with the alternate navigation neural network

## Predictors
### Alt Nav NN
[![Run in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HoverflyHampton/Tetherkit_NN_Utilities/blob/master/Simple%20Alt%20Nav%20NN.ipynb)  
A simple tensorflow network to predict position of the craft given flight characteristics. 

### Random Forest
[![Run in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HoverflyHampton/Tetherkit_NN_Utilities/blob/master/Random_Forest_Alt_Nav.ipynb)  
A random forest desicion tree approach to predictiong the position of the craft given flight characteristics

## Scripts
### nn_file_merge.py
[![Run in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HoverflyHampton/Tetherkit_NN_Utilities/blob/master/Dataset%20Creator.ipynb#scrollTo=80fba116-6099-4737-814a-e2afe6bb57e1)

This script is useful for preproccessing bgu log files to make training data. It takes in a directory of bgu log files, and creates a new csv that contains only the input fields for the nn as well as the calculated true x/y position of the craft. 
#### Requirements
- python 3.9+
- numpy
- pandas
#### Usage
call `python3 nn_file_merge.py -i <INPUT_DIR> -o <OUTPUT_FILE>`
##### Arguments
- INPUT_DIR: The directory of the bgu files to preproccess. Defaults to the current directory
- OUTPUT_FILE: The path to the file to create with all the training data. Defaults to "./hoverfly_nn_test_data.csv"
