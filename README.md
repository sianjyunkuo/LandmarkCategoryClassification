# LandmarkCategoryClassification
DSCI 552 Machine Learning for Data Science group project
## Abstract
This study investigates the impact of pre-processing techniques and freezing layers on model accuracy. The initial model achieved a validation accuracy of 17%, but it was found to be overfitting the data. To address this issue, the group implemented pre-processing techniques on the data and applied freezing layers to prevent the model from overfitting. The pre-processing technique involved using image pre-processing, and the freezing layer method involved freezing the layers. As a result of these techniques, the model's validation accuracy was improved from 17% to 96%. The study provides insights into the benefits of using pre-processing and freezing layers for enhanced model accuracy and performance. This research can be useful for practitioners and researchers interested in improving model performance and reducing overfitting in image identification.

## Introduction
Deep learning has become increasingly important in many fields, such as image recognition, natural language processing, and predictive modeling. However, one common challenge in deep learning is overfitting, where the model becomes too complex and performs well on the training data but poorly on new, unseen data. To address this issue, various techniques have been proposed, including pre-processing techniques and freezing layers.

In this study, we investigate the impact of pre-processing techniques and freezing layers on a model's classification accuracy. Specifically, we focus on image recognition, where pre-processing techniques such as photo pre-processing can be applied to enhance the applicability of data for our own purpose. Additionally, we apply freezing layers to prevent the model from overfitting by restricting the training of certain layers in the network.

The dataset is made up of a selection of pictures of well-known (or lesser-known) sites. A two-level data hierarchy has been used to organize the collection. The categories for the landmarks are on the first level, and the specific landmarks are on the second level. There are 6 categories, including Gothic, Modern, Pagodas, Pyramids, Neoclassical, and Mughal architecture.There are a total of 30 landmarks, 5 for each category. There are 14 photographs for each landmark.

The initial model achieved an accuracy of 17% on validation data accuracy with training accuracy 96%, which was found to be overfitting the data. To improve the validation accuracy and solve the overfitting problem, we implemented pre-processing techniques on the data and applied freezing layers. Our results show that these techniques led to a significant improvement in the model's performance, achieving an accuracy of 96% on a validation set.

This study contributes to the ongoing efforts to improve model accuracy and reduce overfitting in machine learning. The findings can be useful for practitioners and researchers interested in applying pre-processing techniques and freezing layers to improve model performance in image recognition tasks.
