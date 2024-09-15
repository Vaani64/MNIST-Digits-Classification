# MNIST-Digits-Classification-Model

This notebook provides a comprehensive example of classifying handwritten digits using a neural network model on the MNIST dataset. The MNIST dataset consists of 28x28 grayscale images of handwritten digits ranging from 0 to 9. The notebook includes steps for data loading, preprocessing, model creation, training, evaluation, and error analysis.

Key Sections
Imports and Setup:

Essential libraries are imported, including numpy for numerical operations, matplotlib.pyplot for plotting, keras for building and training the neural network, and seaborn for visualizing the confusion matrix. The random seed is set for reproducibility.
Data Loading and Visualization:

The MNIST dataset is loaded and divided into training and test sets. Sample images for each digit class are displayed to provide an overview of the dataset. Labels are converted to one-hot encoded vectors to facilitate multi-class classification.
Data Preparation:

The pixel values of images are normalized to a range of [0, 1] by dividing by 255.0, which helps in stabilizing the training process. Images are reshaped from 28x28 pixels to 784-dimensional vectors to match the input shape expected by the fully connected layers of the model.
Model Creation:

A simple neural network model is constructed using Keras:
Dense Layers: Two hidden layers with 128 units each, using ReLU activation functions.
Dropout Layer: Applied with a rate of 0.25 to reduce overfitting by randomly dropping 25% of neurons during training.
Output Layer: A dense layer with 10 units and a softmax activation function for classifying the digits.
Model Training:

The model is trained on the training dataset using a batch size of 512 and for 10 epochs. The validation set is used to monitor the training process and prevent overfitting.
Model Evaluation:

After training, the model's performance is evaluated on the test dataset. Test loss and accuracy are reported. Predictions are made on test data, and a sample image with its predicted and true label is displayed.
Confusion Matrix:

A confusion matrix is plotted to visualize the performance of the model across different classes, showing the number of true positives, false positives, true negatives, and false negatives for each digit.
Error Analysis:

Errors are analyzed to understand where the model is making mistakes. The notebook identifies the top 5 most significant misclassifications and displays these problematic images along with their predicted and true labels.
This notebook illustrates a fundamental approach to image classification using neural networks, covering all essential steps from data preprocessing to model evaluation and error analysis.

