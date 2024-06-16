Project 4 - Team Purple - Diagnosing Pneumonia through Xray Images.

We are Team Purple and our Team consists of three Team Members. 
1. Khali Baasandorj
2. Michael MacInnis
3. Julian Hahm

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

Now that we understand the challenge, we want to provide some background on dataset.  Our dataset was taken from Kaggle and was selected from restrospective cohorts of pediatric patients of one to five years old from Guangzhou Women & Children's Medical Center.  The Xrays were taken as a part of a patients regular routine care.  These Xrays were screened for quality control and only images that were clear and readable were included in the dataset.  Thereafter, two physicians graded the readable images before allowing them to be part of the dataset. 

<img width="811" alt="Screenshot 2024-06-16 at 8 58 06 AM" src="https://github.com/hakuban-create/Chest_xray_Project_4/assets/154090947/bda99895-26f8-49a4-8723-45496788e7bf">

The dataset is organized into 3 folders.  Train, Test, and Val are the three folders of images and each is divided into two subcategories normal or pneumonia.  The total number of images in the train folder is 5,863.

To start our data processing, we used Google Colab to host the dataset/images and leveraged Python, Numpy, and Matplotlib to help in visualizing our model.  Furthermore, we imported modules to assist with our model.

1. OS - provides functions for creating and removing a directory (folder), fetching its contents, changing and identifying the current directory,
2. cv2 - brings the OpenCV library into the Python script, allowing access to its functions for computer vision and image processing.
3. numpy -  a library that provides a set of high-level functions and features for performing data analysis and manipulation.
4. tensorflow - machine learning
5. tensorflow.keras.preprocessing - adding utilities to work with image data
6. sklearnb preprocessing - provides several common utility functions and transformer classes to change raw feature vectors into a representation that is more suitable for the downstream estimators. I
7. tensorflow.keras. - high-level APIs used to easily train and build models (Keras has built-in Python)
8. io BytesIO - manipulating data in memory for Binary Data

Now that we have the stage set with our tools, we want to quickly walk you through our process.

Data Preprocessing - We used directory_path, train_path, and test_path in Google Colab to work with image data for machine learning. We then checked the number of images in each category (normal vs. pneumonia) for both training and testing sets to verify that our dataset was loaded correctly and that we had the expected number of images in each category.  

 <img width="700" alt="Screenshot 2024-06-16 at 9 41 31 AM" src="https://github.com/hakuban-create/Chest_xray_Project_4/assets/154090947/3e57aaf1-6b13-4cdb-9eba-295660a7b579">


We defined a set of image processing functions to preprocess images for our chest x-ray dataset. These functions include sharpening, denoising, histogram equalization,and resizing.

- The sharpen function applied a sharpening filter to the input image using a specific kernel for sharpening. The kernel we used is a 3x3.
- We used Gaussian blur to help in reducing noise and unwanted details in the image, making it smooth
- Histogram equalization was used to enhance the contrast of an image by spreading out the intensity values across the histogram.
- Resize(image): This function resized the input image to a fixed size of 256x256 pixels using OpenCV's resize function.
- Preprocess(image): This function applied a series of image processing steps to the input image. It first equalized the histogram, then denoised the image, sharpened it, and finally resized it to 256x256 pixels.

Thereafter, we looped through the images in the "NORMAL" category of the training set, reading each image using OpenCV, resizing it to 256x256 pixels, and then appending the resized image to the X_train list while adding the label "0" (indicating "NORMAL") to the y_train list.  Similarly, we processed the images in the "PNEUMONIA" category of the training set. We read each image, resizing it to 256x256 pixels, and then appended it to the resized image to the X_train list while adding the label "1" (indicating "PNEUMONIA") to the y_train list. We did the same for the test dataset and after, we converted our feature and label lists to Numpy Arrays to better suit Machine Learning models.

Section 3 - Michael

Now we are going to talk about our Convolutional neural network (CNN) model for image classification.

A Convolutional Neural Network (CNN) is a type of deep learning model designed specifically for processing structured grids of data. It's particularly effective for tasks involving images, video, and other two-dimensional data.  The key components of a CNN include multiple layers and here is our list below.

<img width="1175" alt="Screenshot 2024-06-16 at 9 54 37 AM" src="https://github.com/hakuban-create/Chest_xray_Project_4/assets/154090947/f8c8bb6e-8398-499b-b44e-872047ec3050">

Input Layer:
- Conv2D layer with 32 filters, a kernel size of (3, 3), and ReLU activation function.
- Input shape of (256, 256, 3) indicating image dimensions of 256x256 pixels with 3 color channels (RGB).

Hidden Layers:
- MaxPooling2D layer with a pool size of (2, 2) for downsampling.
- Flatten the layer to flatten the output for the fully connected layers.
- Additional Conv2D layers with varying numbers of filters and ReLU activation functions.
- MaxPooling2D layers for downsampling after convolutional layers.
- Dense layer with 512 units and ReLU activation function for feature extraction.
- Dropout layer with a dropout rate of 0.5 for regularization.

Output Layer:
- Dense layer with 1 unit and softmax activation function.

Our CNN architecture follows a typical pattern for image classification models, starting with convolutional and pooling layers for feature extraction, followed by fully connected layers for classification.

- Conv2D: This layer defines a convolutional layer with 32 filters, each of size 3x3, using the ReLU activation function. The input shape is set to (256, 256, 3) for images of size 256x256 pixels with 3 color channels (RGB).
- MaxPooling2D: This layer performs max pooling with a pool size of 2x2 to downsample the spatial dimensions.
- Flatten: This layer flattens the input into a 1D array before feeding it into the fully connected layers.
- Dense: This layer consists of 128 neurons with the ReLU activation function for learning complex patterns in the data.
- Dense: The final output layer with 1 neuron and a sigmoid activation function for binary classification tasks.

we also trained our model using the fit method with the training data (X_train and y_train). Here are the parameters:
- epochs=10: This specifies the number of epochs (iterations over the entire training dataset) for training the model.
- shuffle=True: This parameter shuffles the training data before each epoch to prevent the model from memorizing the order of the data.
- verbose=2: This parameter controls the verbosity of the training process, where verbose=2 provides more detailed logging during training.

We also used Keras Tuner to provide flexibility in tuning the model's architecture and hyperparameters. Keras Tuner is a library that helps you find the best hyperparameters for your TensorFlow program. Hyperparameters are variables that control the training process and the topology of a machine learning (ML) model.

- Neurons in the first layer with a range off 5-10 with a step of 2
- Decide the number of hidden layers (up to 3) and the number of neurons in each layer (range 1-10) with step of 2
- Adds a Flatten layer to flatten output from previous layers

Hyperband tuner searches for best hyperparameters for your neural network model by optimizing the validation accuracy over a specified number of epochs and iterations.

![Screenshot 2024-06-16 at 2 06 31 PM](https://github.com/hakuban-create/Chest_xray_Project_4/assets/154090947/b7b38bcc-b5ff-4ff4-9763-858d7bf9984e)

We were able to get our ML model to 85% accuracy.  

Next, we used the model.evaluate() function to return the loss and accuracy of the model on the test data. The loss value represents how well the model is performing, with lower values indicating better performance. The accuracy value shows the proportion of correctly classified instances in the test data.

<img width="751" alt="Screenshot 2024-06-16 at 2 16 13 PM" src="https://github.com/hakuban-create/Chest_xray_Project_4/assets/154090947/56dd05ba-6a9c-4e83-83c0-e5f4fe6c583f">

In addition, we used the predict method to generate predictions from our model and applied a threshold of 0.5 to convert the probabilities into binary predictions. After that, we printed a classification report to evaluate the model's performance on the test data.

The classification_report function provides a summary of important classification metrics such as precision, recall, F1-score, and support for each class.

<img width="864" alt="Screenshot 2024-06-16 at 2 16 20 PM" src="https://github.com/hakuban-create/Chest_xray_Project_4/assets/154090947/361e26cf-040e-4dfb-849f-3e68cd57cf10">

And lastly, we ran a confusion matrix to see correct and incorrect predictions made by the model.  What we noticed is that the model predicts Pnemonia correctly for those who actually have it but misdiagnoses those who are normal to sometimes have Pneumonia.  We believe this to be from the dataset have a much larger volume of Pneumonia images.  

<img width="1108" alt="Screenshot 2024-06-16 at 2 21 01 PM" src="https://github.com/hakuban-create/Chest_xray_Project_4/assets/154090947/4497e16d-0056-449b-a0fb-81879dfb76c6">

We also created a UI that allows us to select an image to see whether or not the model predicts the image to have Pneomonia.  Similar to the confusion matrix we ran, we can see that the model predicts those who have Pneumonia very accurately but something it predicts those who are normal to have Pneumonia. 

Thank you.



Sources & Acknowledgments:

- Xpert Assistant
- ChatGPT
- Data: https://data.mendeley.com/datasets/rscbjbr9sj/2
- Citation: http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5



