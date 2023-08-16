# Indian_Gesture_Reccognition


Indian Sign Language (ISL) is a complete language with its own grammar, syntax, vocabulary and several unique linguistic attributes. It is used by over 5 million deaf people in India. Currently, there is no publicly available dataset on ISL to evaluate Sign Language Recognition (SLR) approaches.
In this work, we present the Indian Lexicon Sign Language Dataset - INCLUDE - an ISL dataset that contains 0.27 million frames across 4,287 videos over 263-word signs from 15 different word categories. INCLUDE is recorded with the help of experienced signers to provide a close resemblance to natural conditions. A subset of 50-word signs is chosen across word categories to define INCLUDE-50 for rapid evaluation of SLR methods with hyperparameter tuning.


#Introduction


Approximately 5% of people around the world suffer from a "disabling" hearing loss that will require rehabilitation. Around 2.42 million Deaf and Mute (D&M) individuals live in India, the world's second most populous country. In modern-day society, people do not think of these impairments as disabilities; it is merely perceived as a lifestyle choice. In reality, life is a lot harder for them as D&M individuals cannot communicate verbally, having to resort to mediums such as using visual media and fingerspelling to communicate. However, the vast majority of non-D&M people are unaware of sign language, resulting in an extensive communication gap. This paper addresses this very issue by proposing a ML model that recognizes the Indian Sign Language (ISL) using Advanced Deep Learning methods like 3D CNN, MoViNet for sign language recognition and a web application that captures these hand gestures and translates them into text. 


#Dataset Description


https://zenodo.org/record/4010759
The INCLUDE dataset has 4292 videos . 
The videos used for training are mentioned in train.csv (3475), while that used for testing are mentioned in test.csv (817 files). 
Each video records 1 ISL sign, signed by deaf students from St. Louis School for the Deaf, Adyar, Chennai.


#Data Preprocessing


Since the Dataset is consists videos that belongs to different classes , we will be taking each video convert it to image and then map the frames  with Labels  using 
keras-vid: library that complements Keras and provides video data handling capabilities. It allows you to load video data, apply transformations, and prepare batches for training 3D CNNs or other video-based models
MoviePy: MoviePy is a Python library built on top of FFmpeg and provides an easy-to-use API for video editing and processing tasks. It can read videos, extract frames, concatenate clips, and perform various video-related operations.


#Model Specifications


3D Convolutional Neural Networks (3D CNNs): 3D CNNs extend traditional 2D CNNs by incorporating the temporal dimension. They can directly process video sequences as spatio-temporal volumes, capturing both spatial and temporal features simultaneously. This makes them well-suited for gesture recognition tasks. Models like C3D (Convolutional 3D) have been commonly used for this purpose.
MoViNet (Mobile Video Networks) is a family of lightweight and efficient deep learning models designed specifically for video understanding tasks, such as action recognition, gesture recognition, and other spatio-temporal analysis tasks. MoViNet models are optimized for deployment on resource-constrained devices like mobile phones, edge devices, or embedded systems.


#Loss Functions


For gesture recognition or action recognition tasks, there are several loss functions and evaluation metrics that you can use to train and assess your models.
Categorical Cross-Entropy (Softmax Loss): This is a standard loss function used for multi-class classification tasks. It is well-suited for gesture recognition, where each gesture corresponds to a specific class label. The softmax loss computes the cross-entropy between the predicted probability distribution and the true one-hot encoded labels.


#Evalluation Rubrics


Accuracy: Accuracy is a widely used metric for classification tasks, including gesture recognition. It measures the percentage of correctly predicted gestures over the total number of samples in the dataset. However, accuracy alone might not be sufficient if your dataset is imbalanced or when different gesture classes have varying importance.

Precision, Recall, and F1-Score: Precision measures the proportion of true positives among all positive predictions, recall measures the proportion of true positives among all actual positive samples, and the F1-score is the harmonic mean of precision and recall. These metrics are useful when there is class imbalance in the dataset.







