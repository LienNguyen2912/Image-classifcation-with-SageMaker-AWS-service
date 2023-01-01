# Image classifcation with SageMaker AWS service
# Introduction
I tried to find a tutorial to guide someone who is very beginner with AWS to create and deploy an image classification model by using the Amazon SageMaker from scratch but I couldn't. Actually, there are many tutorial videos, documents but they require some data which can be downloaded from a URL or knowledge about AWS.</br>
Here is an end-to-end example of using the Amazon SageMaker image classification without any prerequisite steps.

# Prepare the data
In order to make it as simple as possible, we will
- Prepare data on your local PC
- Upload to S3 bucket. Make it public access
- Download the data and transfer it to S3 of the Sagemaker notebook instance for training

## Create training data in RecordIO format
For the training and validation data, we will take the RecordIO file as input.</br>
Prepare your data with the folder structure below on your PC</br>
![data_folders_structure](https://user-images.githubusercontent.com/73010204/210136379-6a6fb0b2-e737-42ef-8097-f68ee4934812.png)</br>
Those images will be converted into RecordIO format using MXNet’s im2rec tool. </br>
After installing **python 3.6**, please do not use the later version of python unless you will get mxnet install error
```sh
py -m pip install --upgrade pip
pip install mxnet
```
Next, we will create .lst and .rec file.</br>
_cd_ to _dataset_dogs_cats_ folder and execute
```sh
# create dataset_rec_train.lst file
python C:/Users/liennt/AppData/Local/Programs/Python/Python36/Lib/site-packages/mxnet/tools/im2rec.py ./dataset_rec_train ./train/ --recursive --list --num-thread 8
# create dataset_rec_train.rec va dataset_rec_train.idx
python C:/Users/liennt/AppData/Local/Programs/Python/Python36/Lib/site-packages/mxnet/tools/im2rec.py ./dataset_rec_train ./train/ --recursive --pass-through --pack-label --num-thread 8
```
Do almost the same to create .lst and .rec file for test folder
```sh
# create dataset_rec_val.lst file
python C:/Users/liennt/AppData/Local/Programs/Python/Python36/Lib/site-packages/mxnet/tools/im2rec.py ./dataset_rec_val ./test/ --recursive --list --num-thread 8
# create dataset_rec_val.lst file
python C:/Users/liennt/AppData/Local/Programs/Python/Python36/Lib/site-packages/mxnet/tools/im2rec.py ./dataset_rec_val ./test --recursive --pass-through --pack-label --no-shuffle --num-thread 8
```
![create_rec_file](https://user-images.githubusercontent.com/73010204/210136380-cf569c68-5093-452f-8f44-f1fbddd089b3.png)</br>
Then your folder will look like</br>
![create_rec_file_done](https://user-images.githubusercontent.com/73010204/210136453-be0044c3-3098-41e0-9aa9-88f2920bd1aa.png)</br>

## Upload to S3 bucket. Make public access

Login your AWS account, go to S3 bucket service and Create a bucket</br>
![createS3](https://user-images.githubusercontent.com/73010204/210160782-eaf0fb55-ba26-4840-9f73-49ee47b003af.png)</br>
The region of the bucket **must be same** with the region of the Sagemaker notebook instance you will create later. We don't need to specifiy where we are living as the region. Actuall, I tried Osaka and then I got error of not support neither _ml.p3.2xlarge_ nor _ml.p2.xlarge_ instance_type when training.</br>
![createS3_2](https://user-images.githubusercontent.com/73010204/210160912-7f4692c1-e428-4a60-8bc0-b7d65a8546f2.png)</br>
![createS3_3](https://user-images.githubusercontent.com/73010204/210160916-b187ffba-72e3-4f55-8e0c-76ef1101f58a.png)</br>
![createS3_4](https://user-images.githubusercontent.com/73010204/210160919-b4033a2c-3146-43c6-9309-1f598ee138e3.png)</br>
Because we want to make this S3 bucket public acess so uncheck the _Block all public access_. We also can do it by edit the _Permission_ later.</br>
But only check off the _Block all public access_ is not enought to make it public accessible, we need to edit the Policy later.</br>
Firstly, let's create the folder structure for our bucket</br>
![createS3_5](https://user-images.githubusercontent.com/73010204/210160966-42f92487-7335-43bd-a6b6-7b79148efffe.png)</br>
Drag and drop the .rec file created above to upload. We will get the S3 bucket as</br>
![createS3_6](https://user-images.githubusercontent.com/73010204/210161196-da7ce4c7-e431-477a-ad42-a50c9521a4c5.png)
