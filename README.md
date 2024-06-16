# Chest_xray_Project_4
We are Team Purple and our Team consists of four members:
1.Khali Baasandorj
2.Michael MacInnis
3.Julian Hahm

### For Project 4, we were tasked to solve, analyze, or visualize a problem using machine learning (ML) with the other technologies we’ve learned.
### We decided to figure out how Machine learning can assist in detecting Pneumonia. 
### Pneumonia is a serious infection or inflammation of the lungs, air sacs, or alveoli, caused by bacteria, viruses, fungi, or chemical irritants. The air sacs fill with pus and other liquid, which can lead to symptoms ranging from mild to severe.  
### Thus, it would be helpful if machine learning could assist in the detection of Pneumonia.  

### Diagnostic evaluation of Pneumonia typically involves chest X-ray imaging, enabling healthcare professionals to better diagnose the type and severity of pneumonia.  When interpreting the x-ray, the radiologist will look for white spots in the lungs (called infiltrates) that identify an infection. This exam will also help determine if you have any complications related to pneumonia such as abscesses or pleural effusions (fluid surrounding the lungs).

# Dataset & Modules
### We used a dataset found on Kaggle that is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal*).

### Train:Contains X-Ray images for training the model.Organized into subfolders for each image category (Pneumonia/Normal).Total of X-Ray images: 5,863.

### Test:Contains X-Ray images for testing the trained model.Organized into subfolders for Pneumonia and Normal categories.

### Validation (Val):Holds X-Ray images for validation purposes.Subfolders correspond to the Pneumonia and Normal categories as well.

### We also had to import modules to help manipulate and process the dataset to ensure our Machine Learning model would be able to help diagnose images with Pneumonia at 85 percent plus accuracy.  We imported the following modules to assist with manipulating the dataset.

#### OS
#### cv2
#### google colab
#### drive
#### numpy
#### tensorflow
#### sklearnb preprocessing
#### tensorflow.keras.models
#### tensorflow.keras.utils
#### tensorflow.keras.layers
#### tensorflow.keras.preprocessing 
#### io BytesIO

# Data Preprocessing 
### We used directory_path, train_path, and test_path in Google Colab to work with image data for machine learning. We then checked the number of images in each category (normal vs. pneumonia) for both training and testing sets to verify that our dataset was loaded correctly and that we had the expected number of images in each category.  
### After, we looped through the images in the "NORMAL" category of the training set, reading each image using OpenCV, resizing it to 256x256 pixels, and then appending the resized image to the X_train list while adding the label "0" (indicating "NORMAL") to the y_train list.
### Similary, we processed the images in the "PNEUMONIA" category of the training set. We read each image, resizing it to 256x256 pixels, and then appended it to the resized image to the X_train list while adding the label "1" (indicating "PNEUMONIA") to the y_train list.
### we did the same for the test dataset and after, we converted our feature and label lists to Numpy Arrays to better suite Machine Learning models.
### We then verified our shapes from the testing and training arrays were correctly processed for model training. 

# The Model
### We used a Convolutional Neural Network (CNN) model using Keras Sequential API for image classification tasks. Here's a breakdown of the layers in our model

## Conv2D: This layer defines a convolutional layer with 32 filters, each of size 3x3, using the ReLU activation function. The input shape is set to (256, 256, 3) for images of size 256x256 pixels with 3 color channels (RGB).
## MaxPooling2D: This layer performs max pooling with a pool size of 2x2 to downsample the spatial dimensions.
## Flatten: This layer flattens the input into a 1D array before feeding it into the fully connected layers.
## Dense: This layer consists of 128 neurons with the ReLU activation function for learning complex patterns in the data.
## Dense: The final output layer with 1 neuron and a sigmoid activation function for binary classification tasks.

### we also trained our model using the fit method with the training data (X_train and y_train). Here are the parameters:
## epochs=10: This specifies the number of epochs (iterations over the entire training dataset) for training the model.
## shuffle=True: This parameter shuffles the training data before each epoch to prevent the model from memorizing the order of the data.
## verbose=2: This parameter controls the verbosity of the training process, where verbose=2 provides more detailed logging during training.







