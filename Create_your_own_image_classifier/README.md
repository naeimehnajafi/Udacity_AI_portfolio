# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

## Image Classifier Project

This project is a Jupyter Notebook that shows how to train an image classifier using deep learning in Python. The classifier is trained on a dataset of 102 different species of flowers, which can be used in a mobile application, for example, to recognize the name of the flower in the camera.

The project is broken down into three parts:

1. Load and preprocess the image dataset
2. Train the image classifier on your dataset
3. Use the trained classifier to predict image content

To complete this project, the following packages must be imported:

- matplotlib
- numpy
- torch
- torchvision
- PIL
- json
- collections

The data is loaded using the `torchvision` package, and it is split into three parts: training, validation, and testing. The training set is transformed with random scaling, cropping, and flipping to improve the network's performance. The validation and testing sets are not transformed, but the images are resized and cropped to the appropriate size.

The pre-trained networks used in this project were trained on the ImageNet dataset, and each color channel was normalized separately. For all three sets, the means and standard deviations of the images are normalized to what the network expects: means are [0.485, 0.456, 0.406], and standard deviations are [0.229, 0.224, 0.225], calculated from the ImageNet images.

In addition, a mapping from category label to category name is loaded in the project.

After the classifier is trained, it can be exported for use in other applications. The network in this project learns about flowers and ends up as a command line application. This project can be used as a starting point to build other applications, such as an app that takes a picture of a car and tells you the make and model.

