# Skin-lesion-classification-using-machine-learning-and-deep-learning-algorithms.

**ABSTRACT**

Dermatology is one field where deep learning architectures for image classification have shown amazing results. Here are a few challenges in skin lesion classification, such as Melanoma. The skin lesion image has various similarities, which leads to inaccurate classification. Even though dermatology professionals only diagnose 75 to 80% of the time. Our goal in this study is to use deep learning models and machine learning models such as Support vector machine, Logistic regression, stacking classifier, random forest, Convolutional neural network, and transfer learning algorithms to classify skin lesions into seven groups such as Melanoma, benign keratosis, and so on. The Ham10000 dataset was downloaded from the official website to carry out this research. First, data augmentation was performed in data pre-processing to create several versions of the photos by rotating, flipping, cropping, and zooming. Following that, models are evaluated using previously unseen data. The results show that the ResNet50v2 algorithm outperforms all other models.

**1.2	Aim **

This study aims to classify 4 types of skin lesion problems, such as Melanoma, Benign keratosis, and basal cell carcinoma (bcc), using machine learning and deep learning algorithms.

**1.3 Objectives**

The objectives of this study are given down below.
•	To employ feature extraction techniques like Histogram of oriented Gradient, convolutional neural network and transfer learning algorithm.
•	To implement various machine learning and deep learning models like logistic regression, support vector machine, stacking, random forest classifier, convolutional neural networks, and transfer learning algorithms.
•	To apply hyperparameter tuning to find the optimal parameters of the machine learning algorithm. 
•	To evaluate the model's performance using precision, recall, f1 score and accuracy.


**1.4 Plan**

•	The first step in this research was to collect the data from Harvard dataverse. It contains seven types of skin lesion problems such as Melanoma, Benign keratosis, basal cell carcinoma (bcc), Actinic keratoses and intraepithelial carcinoma (akiec), Dermatofibroma (df), and vascular lesions (vasc). For this experiment, only four classes are selected. 
•	The second step is to all download and install all the necessary software and libraries for this research experiment. Then the dataset is loaded with the help of the Pandas library. 
•	The third step is to perform data analysis to find the distribution of each class and plot the image, respectively, using the matplotlib and seaborn library. The following stage was to extract features from the image using the Histogram of Oriented Gradient, after which machine learning methods were applied, and the model was tested on the test set.
• apply Image processing using the Image data generator function. After that, CNN and the transfer learning algorithm are used as feature extractors to extract the essential components from the image. Next, the deep learning algorithm is applied to train the model and then it is evaluated on the test set using the classification metrics like precision, recall, f1 score and accuracy.
•	The NEXT step is to compare all the algorithms to determine which algorithm performs best among all in classifying the skin lesion problems. 


