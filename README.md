# Kitchenware Classification Project

![Imgur](https://i.imgur.com/Q5NNJTE.jpg)
*Collage created from training images using Python Image Library PIL with the code collage.py in this repo*

This repository contains the capstone project carried out as part of online course [Machine Learning Zoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp) designed and instructed by [Alexey Grigorev](https://github.com/alexeygrigorev) and his team from [DataTalks.Club](https://datatalks.club/). This project was 2 weeks long and peer-reviewed. Idea of this project is to implement everything that we have learnt in the last 11 weeks of courseworks.

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

Let us imagine a use case where you are running an e-commerce platform selling kitchenware prodcuts. A user visits your online plaform and wants to sell a kitchenware through your website. He uploads an image of his item into your website. 

We would like to improve user experience and orgnaization of items in your website by implementing an image classification system that can automatically categorize the kitchenware product uploaded by the user into relevant item such a spoon, a fork, a plate etc. User will upload the image and classification system will automatically tell whether it is a spoon or a fork or something else. This will allow the user to list his item in the website more easily . 

To solve this problem, we'll develop a deep learning model that can accurately classify kitchenware images into the appropriate categories. This will require collecting and labeling a large dataset of kitchenware images, and training a model on this dataset to learn the visual features and patterns that distinguish different types of kitchenwares. Thankfully a large sets of kitchenware data have already been colloected by Alexey Grigorev and is made available for everyone on kaggle. We'll use this dataset to train our model. 

Once we have a trained model, we will need to deploy it in the cloud , so that it can later be integrated into website and finally user can easily upload their items. 

## 2. About the Dataset

You can get the dataset from [kaggle](https://www.kaggle.com/competitions/kitchenware-classification/data). 

For getting the dataset, you may have to first accept the rules of the competition. Since it is a large dataset, I haven't hosted it on this git-hub repo.

Datset contains images of different kitchenware. Images are of following  six classes:

- cups
- glasses
- plates
- spoons
- forks
- knives

Following folders and files inside `dataset` folder are useful to us:

- images - all images of kitchenware in JPEG format
- train.csv - list of image IDs of all images in `images` folder and class of that particular image

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
Model is deployed in the cloud with AWS Lambda and API Gateway.

## 4. About files in this repo


├── collage.py<br>
├── Dockerfile<br>
├── environment.yml<br>
├── kitchenware-model.h5<br>
├── kitchenware-model.tflite<br>
├── lambda_function.py<br>
├── LICENSE<br>
├── notebook.ipynb<br>
├── notebook-serverless-deployment.ipynb<br>
├── README.md<br>
├── test-image.jpg<br>
├── test.py<br>
├── train.py<br>
└── xception_v4_2_16_0.965.h5<br>

0 directories, 14 files


Below is a description of the key files:

|  File name |      Description       |
|:--------:|:-----------------------------------:|
|    **README.md**   |  The file you are reading now, meant for the user as an introduction to help navigating the project| 
|    **notebook.ipynb**   |  Jupyter notebook file where EDA, training models, parameter tuning etc. are done during development in Saturn Cloud|
|    **train.py**   |  python script converted from `notebook.ipynb` |
|    **xception_v4_2_16_0.965.h5**   |  Best performing model saved from `notebook.ipynb` |
|    **kitchenware-model.h5**   |  Same model as above, just downloaded from github and name changed |
|    **kitchenware-model.tflite**   |  tensorflow lite version of above tensor flow model |
|    **lamda_function.py**   |  script containing lambda function |
|    **Dockerfile**   |  Dockerfile for building dokcer image |
|    **test.py**   |  python script used for testing locally and also as webservice with API Gateway |
|    **notebook-serverless-deployment.ipynb**   |  Jupyter notebook file for testing model locally|
|    **environment.yml**   |  environment file for conda environment used for testing model locally |
|    **test-image.jpg**   |  A random test image downloaded from internet |
|    **collage.py**   |  a simple fun python script for creating a collage of images used in this readme.md|

## 5. Development System

**Model development:** 
Saturn Cloud

**Model deployment (for testing locally):**
OS: Ubuntu 18.04.6 LTS<br>
Architecture: x86_64<br>
conda virtual environment for development<br>

## 6. How to reproduce this project
### 6.1 Development

Development part of this project i.e. getting the best performing and saving it, basicall everything in `notebook.ipynb` is done in Saturn Cloud Jupyter otebook server with GPU. 

If you don't have an account yet, you can do it with this [link](https://saturncloud.io/?utm_source=Youtube+&utm_medium=YouTube&utm_campaign=AlexeyGMLZoomCamp&utm_term=AlexeyGMLZoomCamp) 

Make sure that there is an `utm` in this link. Then you get a 30 hours of free GPU hours and can run this notebook. 

Then click on the below Run in Saturn Cloud:

[![Run in Saturn Cloud](https://saturncloud.io/images/embed/run-in-saturn-cloud.svg)](https://app.community.saturnenterprise.io/dash/o/community/resources?templateId=edc9880a27cc4593862bbdb872f98023)

This will allow you to clone my repo.

### 6.2 Deployment using tensorflow-lite
After development, best model is downloaded to local folder and first deployed locally. This is done in the `notebook-serverless-deployment.ipynb` file. It is done in a local Anaconda environment.

For this, you first have to install Anaconda on your system, if you have not done it already. Install it by following these instructions in this [Anaconda](https://www.anaconda.com/products/distribution) page. This Site automatically detects the operating system and suggest the correct package.

I have created a `environment.yml` dependency file by running the command `conda env export > environment.yml` inside my activated conda envirnment. You can now recreate this environment with the command:

```
conda env create -f environment.yml
``` 
You can check if the environment `ml-zoomcamp` is created by listing all the conda environment available with the command:

```
conda info --envs
```
Activate the environment with:

```
conda activate ml-zoomcamp
```

Now you should be able to run `notebook-serverless-deployment.ipynb`, run it and play with it.

This notebook is then converted into a script `lamda-function.py`. A docker file is built to use it for lamda function. 

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
  {
    "repository": {
        "repositoryArn": "arn:aws:ecr:us-east-1:546575206078:repository/kitchenware-tflite-images",
        "registryId": "546575206078",
        "repositoryName": "kitchenware-tflite-images",
        "repositoryUri": "546575206078.dkr.ecr.us-east-1.amazonaws.com/kitchenware-tflite-images",
        "createdAt": 1671572784.0,
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
base) bsarma@turing:~/GitHub/kitchenware-classification-project$ docker tag kitchenware-model:latest $REMOTE_URI

(base) bsarma@turing:~/GitHub/kitchenware-classification-project$ docker push $REMOTE_URI

```

Now we can use this ECR for lambda function. We clicked previously on `Container image` as in figure above. Then  

This was about deploying our lambda function using aws.  

Now, we want to use it as webservice.  For this, we'll expose the lamda function we created as a web service using API Gateway, aservice from AWS. 

- Demonstration of deployment to AWS Lambda: [https://youtu.be/r7pxXrNbN8M](https://youtu.be/r7pxXrNbN8M)

- Demonstration of API-Gateway: [https://youtu.be/-SPoIwNjBbI](https://youtu.be/-SPoIwNjBbI)
	
## 7. Conclusions

**Decelopment**

* An accuracy of 0.965 is obtained on test data without any overfitting
* Best model is saved and deployed using AWS Lambda and API Gateway
* Limited EDA on images are done
* Model is trained on multiple variations of neural network: with dropout and without dropout, with an extra inner dense layer and without
* parameter tuning is done for learning rate, dropout rate, size of the extra inner layer etc. 

**Deployment**

* Model is deployed with AWS Lambda and API Gateway

## 8. References

- [Github repository of the course Machine Learning Zoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp) 
- [Youtube Playlist where course videos are hosted](https://www.youtube.com/playlist?list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR)



>work done after submitting this capstone project on 21.12.2022 with Commoit ID `1f283a8`:
1. cleaned train.py file
2. corrected appearance of files and folders section in readme.md
3. corrected appearance of Model deployment (for testing locally) part under 5. Development System section in readme.md

## Contacts
If you face any problem in running any part of the project: 

- contact me at `b.sarma1729[AT]gmail.com` or,

- dm on DataTalks.Club slack `@Bhaskar Sarma`.

Last but not the least, if you like the work, consider clicking on the ⭐