# Topo-CNN
The Topo-Ret repository includes three distinct code components for the classification of retinal images.

# Feature_Extraction.py
The Feature_Extraction.py code comprises functions for extracting topological features (Betti-0 and Betti-1) from retinal images, including both grayscale and RGB channels.

# Machine_Learning.py
The Machine_Learning.py code is dedicated to various machine learning tasks, such as XGBoost, Random Forest (RF). It is used to classify vectors representing Betti-0 and Betti-1 features obtained from Feature_Extraction.py.

# CNN_+_TDA.py
The CNN_+_TDA.py code combines features derived from Topological Data Analysis (TDA), which are obtained from Feature_Extraction.py, and Convolutional Neural Network (CNN) features from pretrained models.
