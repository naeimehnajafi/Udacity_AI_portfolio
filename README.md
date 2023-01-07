# Image Classification for a City Dog Show

## Project Goal

The goal of this project is to improve my programming skills using Python. This is the first project towards the Udacity nanodegree program on AI programming with Python. In this project, I will use a created image classifier to identify dog breeds. The focus is on Python and not on the actual classifier. For the second project, I will be designing the image classifier myself.

## Description

My city is hosting a citywide dog show and I have volunteered to help the organizing committee with contestant registration. Every participant that registers must submit an image of their dog along with biographical information about their dog. The registration system tags the images based upon the biographical information.

However, some people are planning on registering pets that aren’t actual dogs. I need to use an already developed Python classifier to make sure the participants are dogs. 
## My Tasks

Using my Python skills, I will:

- Determine which image classification algorithm works the "best" on classifying images as "dogs" or "not dogs".
- Determine how well the "best" classification algorithm works on correctly identifying a dog's breed. 
- Time how long each algorithm takes to solve the classification problem. 

**Note:** With computational tasks, there is often a trade-off between accuracy and runtime. The more accurate an algorithm, the higher the likelihood that it will take more time to run and use more computational resources to run. 

## Important Notes

For this image classification task, I will be using an image classification application using a deep learning model called a convolutional neural network (often abbreviated as CNN). CNNs work particularly well for detecting features in images like colors, textures, and edges; then using these features to identify objects in the images. I'll use a CNN that has already learned the features from a giant dataset of 1.2 million images called ImageNet. There are different types of CNNs that have different structures (architectures) that work better or worse depending on your criteria. With this project, I'll explore the three different architectures (AlexNet, VGG, and ResNet) and determine which is best for my application.

A classifier function in classifier.py will allow me to use CNNs to classify my images. The test_classifier.py file contains an example program that demonstrates how to use the classifier function. For this project, I will be focusing on using my Python skills to complete these tasks using the classifier function.

Certain breeds of dogs look very similar. The more images of two similar-looking dog breeds that the algorithm has learned from, the more likely the algorithm will be able to distinguish between those two breeds. We have found the following breeds to look very similar: Great Pyrenees and Kuvasz, German Shepherd and Malinois, Beagle and Walker Hound, amongst others.

## Principal Objectives

1. Correctly identify which pet images are of dogs (even if the breed is misclassified) and which pet images aren't of dogs.
2. Correctly classify the breed of dog, for the images that are of dogs.
3. Determine which CNN model architecture (ResNet, AlexNet, or VGG), "best" achieve objectives 1 and 2.
4. Consider the time resources required to best achieve objectives 1 and 2, and determine if an alternative solution would have given a "good enough" result, given the amount of time each of the algorithms takes to run.

## TODO

- Edit program check_images.py

The check_images.py is the program file that I will be editing to achieve the four objectives above. This file contains a main() function that outlines how to complete this program through using functions that have not yet been defined. I will be creating these undefined functions in check_images.py to achieve the objectives above.

All of the TODOs are listed in check_images.py

