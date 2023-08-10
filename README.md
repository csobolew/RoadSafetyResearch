# RoadSafetyResearch

To run:  
  
1. Install requirements from requirements.txt  
2. Run using Python 3.8.0  
3. Relies on images 0-4649 in result/ folder
4. Also uses .npz file for training dataset labels
5. Also relies on labels npy files in labels/ folder and modelcropboth3.pth for inverted image model

The models:
The models are both trained using ResNet50 as the base with a FC network on top consisting of two linear layers, one that is 2048 to 128 and one that is 128 to 2.
The standard ResNet50 normalization is applied to the inputs, along with resizing the images to the proper size and applying RandomAffine and RandomHorizontalFlip.
Images are cropped to the area with the car before training using either the darkest or lightest points on the image (in the paticular area where the car is typically located).
Afterwards, there are 8 epochs of training using either the normal or inverted data depending on the model.
To determine which model is applied to a given image, a median criterion is applied, and depending on the result, a model is picked to sample from.
