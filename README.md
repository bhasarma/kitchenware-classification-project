# kitchenware-classification-project

![Imgur](https://i.imgur.com/Q5NNJTE.jpg)
*Collage created using Python Image Library PIL with the code collage.py in this repo*

This repository contains the capstone project carried out as part of online course [Machine Learning Zoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp) designed and instructed by [Alexey Grigorev](https://github.com/alexeygrigorev) and his team from [DataTalks.Club](https://datatalks.club/). This project is 2 weeks long and peer-reviewed. Idea of this project is to implement everything that we have learnt in the last 11 weeks of courseworks.

## Table of Contents:
1. Business Problem Description
2. About the Dataset
3. Approach to solve the problem
	3.1 EDA to understand dataset
	3.2 Model training
	3.3 Model Deployment in the cloud
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
### 3.1 EDA to understand dataset
When it comes to image dataset, there are not much necessity for EDA unlike tabular data. I have done the following EDA for this dataset:

- number of images and classes
- name of classes and their frequencies
- random visualization of few images
- distribution of image size (kB) and resolution (width and height pixels)

 
### 3.2 Model training
Model is trained with transfer learning while using the CNN model Xception from keras applications as base model. Model for our images is trained on the top of it. Parameter tuning is done for epochs and learning rate. 

### 3.3 Model Deployment in the cloud

## 4. About files and folders in this repo
|  File name |      Description       |
|:--------:|:-----------------------------------:|
|    **README.md**   |  The file you are reading now, meant for the user as an introduction to help navigating the project| 
|    **notebook.ipynb**   |  Jupyter notebook file where EDA, training models, parameter tuning etc. are done during development in Saturn Cloud|
|    **notebook-serverless-deployment.ipynb**   |  Jupyter notebook file for deployment ran locally|
|    **collage.py**   |  a small fun python script for creating a collage of images used in this readme.md|

## 5. Development System

## 6. How to reproduce this project
### 6.1 Development
### 6.2 Deployment using tensorflow-lite

- converting tensorflow / keras model to TF-Lite format
- removing tensorflow dependency
- converting above notebook into a python script with `nbconvert`
- creating a lamda function in the code `lamda_function.py`
- packaging everything in a docker container for uploading to AWS Lambda
	- build it localy with `docker build -t kitchenware-model .`
	- run (test) locally with `docker run -it --rm -p 8080:8080 kitchenware-model:latest`
	- test on another terminal with `python test.py`

- creating the lamda function
	- go to lamda on AWS
	![Imgur](https://i.imgur.com/OL0fEcn.png)
	
	
	- *click on create function*
	![Imgur](https://i.imgur.com/UlKkR22.png)
	
	- select container image
	![Imgur](https://i.imgur.com/5oZ8kTa.png)
	
	Now we need to publish the docker image we created into Amazon ECR (Elastic Container Repository). 
	- Go to AWS ECR
	![Imgur](https://i.imgur.com/bBDofxf.png) 
	
	Now, we'll add docker image to ECR using command line utility. Everything can be done through the web interface as well. 
	
1.Install awscli with the following command, if you don't have it installed already:

```bash
(base) bsarma@turing:~$ pip install awscli
``` 

2.configure awscli for the first time with:

```bash
(base) bsarma@turing:~$ aws configure
AWS Access Key ID [None]: ACCESSKEY
AWS Secret Access Key [None]: SECRETACESSKEY
Default region name [None]: us-east-1
Default output format [None]: 
```
3.create AWS Elastic Container Repository 

```bash
(base) bsarma@turing:~$ aws ecr create-repository --repository-name kitchenware-tflite-images
{
    "repository": {
        "repositoryArn": "arn:aws:ecr:us-east-1:5421152122142:repository/kitchenware-tflite-images",
        "registryId": "5421152122142",
        "repositoryName": "clothing-tflite-images",
        "repositoryUri": "5421152122142.dkr.ecr.us-east-1.amazonaws.com/clothing-tflite-images",
        "createdAt": 1669465123.0,
        "imageTagMutability": "MUTABLE",
        "imageScanningConfiguration": {
            "scanOnPush": false
        },
        "encryptionConfiguration": {
            "encryptionType": "AES256"
        }
    }
}

```
**4. Publish the image just created**
4.1 first log into the registry wtih
```bash
(base) bsarma@turing:~$ aws ecr get-login --no-include-email | sed 's/[0-9a-zA-Z=]\{20,}/PASSWORD/g'

docker login -u AWS -p PASSWORD https://546575206078.dkr.ecr.us-east-1.amazonaws.com
```

`sed` used above is a command line utility in linux that allows us to do different text manipulations including regular expressions. 

The output of the above command is something we want to execute. Then we'll be able to login to the registry above with docker. Then we'll be able to push to this registry. So we want to execute whatever the above command returns. For this, we'll put the above command inside a brace and followed by a `$`.

```bash
(base) bsarma@turing:~$ $(aws ecr get-login --no-include-email)
WARNING! Using --password via the CLI is insecure. Use --password-stdin.
WARNING! Your password will be stored unencrypted in /home/bsarma/.docker/config.json.
Configure a credential helper to remove this warning. See
https://docs.docker.com/engine/reference/commandline/login/#credentials-store

Login Succeeded
```
```bash
$ export REMOTE_URI=546575206078.dkr.ecr.us-east-1.amazonaws.com/kitchenware-tflite-images:kitchenware-model-xception-v4-001
$ echo $REMOTE_URI
546575206078.dkr.ecr.us-east-1.amazonaws.com/kitchenware-tflite-images:clothing-model-xception-v4-001
```

```bash
base) bsarma@turing:~/GitHub/kitchenware-classification-project$ docker tag clothing-model:latest $REMOTE_URI

(base) bsarma@turing:~/GitHub/kitchenware-classification-project$ docker push $REMOTE_URI

```
This was about deploying our lambda function using aws.  

Now, we want to use it as webservice.  For this, we'll expose the lamda function we created as a web service. 


	
## 7. Conclusions

## 8. References

- [Github repository of the course Machine Learning Zoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp) 
- [Youtube Playlist where course videos are hosted](https://www.youtube.com/playlist?list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR)

## Contacts
If you face any problem in running any part of the project: 

- contact me at `b.sarma1729[AT]gmail.com` or,

- dm on DataTalks.Club slack `@Bhaskar Sarma`.

Last but not the least, if you like the work, consider clicking on the ⭐