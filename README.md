# kitchenware-classification-project

![Imgur](https://i.imgur.com/Q5NNJTE.jpg)
*Collage created using Python Image Library PIL with the code collage.py in this repo*

This repository contains the capstone project carried out as part of online course [Machine Learning Zoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp) designed and instructed by [Alexey Grigorev](https://github.com/alexeygrigorev) and his team from [DataTalks.Club](https://datatalks.club/). This project is 2 weeks long and peer-reviewed. Idea of this project is to implement everything that we have learnt in the last 11 weeks of courseworks.

## Table of Contents:
1. Business Problem Description
2. About the Dataset
3. Approach to solve the problem
4. About files and folders in this repo
5. Development System
6. How to reproduce this project
7. Conclusions
8. References


## 1. Business Problem Description
Kitchenware refers to a variety of products that are used in the kitchen for cooking, cutting vegetables, baking, eating etc. These products can include cups, glasses, plates, spoons, forks and kinves. 

Let us imagine a use case where you are the Head of Data for an e-commerce giant selling kitchenware prodcuts. A user visits your online plaform and wants to sell one of the mentioned items through your website. He uploads an image of his item into your website. 

We would like to improve user experience and orgnaization of items in your website by implementing an image classification system that can automatically categorize the kitchenware product uploaded by the user into relevant item such a spoon, a fork, a plate etc. User will upload the image and classification system will automatically tell whether it is a spoon or a fork or something else. This will allow the user to list his item in the website more easily . 

To solve this problem, we'll develop a deep learning model that can accurately classify kitchenware images into the appropriate categories. This will require collecting and labeling a large dataset of kitchenware images, and training a model on this dataset to learn the visual features and patterns that distinguish different types of kitchenwares. Thankfully a large sets of kitchenware data have already been colloected by Alexey Grigorev and is made available for everyone on kaggle. We'll use this dataset to train our model. 

Once we have a trained model, we will need to deploy it in the cloud , so that it can later be integrated into website and finally user can easily upload their items. 

## 2. About the Dataset

You can get the dataset from [kaggle](https://www.kaggle.com/competitions/kitchenware-classification/data). 

Datset contains the images of different kitchenware. Images are of six classes:

- cups
- glasses
- plates
- spoons
- forks
- knives

Dataset has the following files in it:

- images - all the kitchenware images in the JPEG format
- train.csv - Image IDs and class of that particular image for training data
- test.csv - Image IDs of images for test data
- sample_submission.csv - a sample submission file in the correct format 

Out of the above files, only `images` and `train.csv` are useful for us. sample_submission.csv is a file meant only for the kaggle compition. test.csv is a list of images with their IDs. This is also useful only for kaggle compition format, but not for us, since we don't know the labels of these images.

We'll start with images in `train.csv`, then split these images into train, validation and test dataset. 

## 3. Approach to solve the problem

## 4. About files and folders in this repo

## 5. Development System

## 6. How to reproduce this project

## 7. Conclusions

## 8. References

- [Github repository of the course Machine Learning Zoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp) 
- [Youtube Playlist where course videos are hosted](https://www.youtube.com/playlist?list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR)

## Contacts
If you face any problem in running any part of the project: 

- contact me at `b.sarma1729[AT]gmail.com` or,

- dm on DataTalks.Club slack `@Bhaskar Sarma`.

Last but not the least, if you like the work, consider clicking on the ⭐