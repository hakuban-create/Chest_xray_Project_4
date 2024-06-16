# Chest_xray_Project_4
We are Team Purple and our Team consists of four members:
1.Khali Baasandorj
2.Michael MacInnis
3.Julian Hahm

Section 1 - Julian

For Project 4, we were tasked to solve, analyze, or visualize a problem using machine learning (ML) with the other technologies we’ve learned.

We chose to see how Machine learning can assist in detecting Pneumonia through scanning Xray images.  Before we dive into what we did, let's first start with some background on Pneumonia.

Pneumonia is a prevalent respiratory infection that affects people of all ages worldwide, with significant morbidity and mortality rates, particularly among high-risk populations. Effective vaccination, early diagnosis, and appropriate treatment are essential in reducing the impact of pneumonia on public health as it is a condition that affects millions of people each year.

Pneumonia is a serious infection or inflammation of the lungs, air sacs, or alveoli, caused by bacteria, viruses, fungi, or chemical irritants. The air sacs fill with pus and other liquid, which can lead to symptoms ranging from mild to severe.  X ray images can be crucial in helping diagnosing Pneumonia by visualizing the condition of the lungs.  Listed below are a some but not all ways that Xray imaging helps diagnose Pneumonia.

1.  Identification of Infiltrates: Pneumonia typically causes inflammation and fluid buildup (consolidation) in the air sacs (alveoli) of the lungs. On an X-ray, this appears as white or opaque areas that indicate the presence of fluid or tissue density changes.

2. Location and Extent: X-rays can show the affected areas of the lungs and help determine whether pneumonia is localized to one area (lobar pneumonia) or spread throughout both lungs (bronchopneumonia).

3. Comparison with Healthy Tissue: X-rays allow comparison between affected and unaffected areas of the lungs. This contrast helps in identifying abnormalities and assessing the severity of pneumonia.

![Screenshot 2024-06-11 at 6 57 19 PM](https://github.com/hakuban-create/Chest_xray_Project_4/assets/154090947/0dffd650-9236-4874-b080-6880698e6d6d)

In the image above you can see how Xrays are used to visualize whether there may be Pneumonia.
![Screenshot 2024-06-16 at 8 37 47 AM](https://github.com/hakuban-create/Chest_xray_Project_4/assets/154090947/a85b0dd8-69d9-4d46-aa46-e21665fb4235)


Left -  normal chest X-ray depicts clear lungs without any areas of abnormal opacification in the image.
Middle - Bacterial pneumonia (middle) typically exhibits a focal lobar consolidation, in this case in the right upper lobe (white arrows)
Right - Viral pneumonia (right) manifests with a more diffuse ‘‘interstitial’’ pattern in both lungs.

<img width="568" alt="Screenshot 2024-06-16 at 8 55 22 AM" src="https://github.com/hakuban-create/Chest_xray_Project_4/assets/154090947/6c5e8d28-11c1-4686-945e-c24f2a5cd26e">

Here is another image showing the fluid and consolidations found in the right lung.




Section #2 - Khali

Now that we understand the challenge, we want to give some background on dataset.  Our dataset was taken from Kaggle and were selected from restrospective cohorts of pediatric patients of one to five years old from Guangzhou Women & Children's Medical Center.  The Xrays were taken as a part of a patients regular routine care.  These Xrays were screened for quality control and only images that were clear and readable were included in the dataset.  Thereafter two physicians graded the readable images before allowing them to be part of the dataset. 

<img width="811" alt="Screenshot 2024-06-16 at 8 58 06 AM" src="https://github.com/hakuban-create/Chest_xray_Project_4/assets/154090947/bda99895-26f8-49a4-8723-45496788e7bf">


The dataset is organized into 3 folders.  Train, Test and Val are the three folders of images and each are divided into two subcategories of normal or pneumonia.  The total number of images within our dataset is 5,863.


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



Sources & Acknowledgments:

Xpert Assistant

ChatGPT

Data: https://data.mendeley.com/datasets/rscbjbr9sj/2
License: CC BY 4.0
Citation: http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5



