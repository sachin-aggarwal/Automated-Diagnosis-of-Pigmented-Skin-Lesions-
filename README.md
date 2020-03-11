INTRODUCTION: 
 
In this era of Homo Economicus, every country is exploiting resources for growth which is the cause of environmental pollution. These particulates cause damage to the skin resulting in cancer(Melanoma) or tumor(Benign Keratosis). Moreover, depletion of the ozone layer due to air pollution results in larger portion of UV rays entering the earth which is carcinogenic in nature.  
 
Although benign keratosis results in malignant looking skin, it is not cancerous. Whereas, melanoma is. Thus it is a challenging task to identify whether that portion of skin is affected by cancer or not. 
 
 Other organs like lungs, eyes, etc. are also affected by the pollutants but skin being the outermost organ continuously suffers and shows symptoms to the naked eye. So, it is easier to take images of the affected area. And hence can be analysed and predicted by using computers.


Dataset : 

The dataset consists of 1067 training images and 46 testing images of Melanoma cells and 1011 training and 88 testing images for Benign Keratosis. 
 
 
Proposed System :

In this study, we applied an end-to-end CNN framework (machine learning system) to detect malignant lesions using images from ISIC archive dataset. We applied different types of optimization and selected the best combination of fine-tuned CNNs.We built a 26-layer model which contains eleven convolutional (Conv2D), five max-pooling (MaxPool2D)and four fully connected layers. The input image is the first layer (h × w × d which h × w is the pixel size and d is the color channel, here is 128 × 128 × 3). For applying deep-learning, we have utilized Keras which is a deep learning framework for implementing CNN . Libraries of python were used for image processing, model building and graph plotting. 
 
 
 CONCLUSION :
 
 Average Accuracy of the Model = ((0.67+0.87)/2)*100 =​ 77% 
 Thus, we can predict whether the affected area is cancerous or not with an accuracy of ​77%​. 
 Although further data analysis is necessary to improve its accuracy, CNN would be helpful for the classication of diseases and in particular for the early detection of skin cancers. Analysis of the results obtained by testing an ISIC dataset  suggests that using our case-based system for representation of cases via CNN is t for the purpose of supporting users by providing relevant information  related to each disease. Further work will involve extending the training phase by using more images and normalization methods to improve the performance of our system and increase the accuracy 
