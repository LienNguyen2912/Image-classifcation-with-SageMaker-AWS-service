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
### Create a S3 bucket
Login your AWS account, go to S3 bucket service and Create a bucket</br>
![createS3](https://user-images.githubusercontent.com/73010204/210160782-eaf0fb55-ba26-4840-9f73-49ee47b003af.png)</br>
The region of the bucket **must be same** with the region of the Sagemaker notebook instance you will create later. We don't need to specifiy where we are living as the region. Actually, I tried Osaka and then I got error of not support neither _ml.p3.2xlarge_ nor _ml.p2.xlarge_ instance_type when training.</br>
Fill your bucket name</br>
![createS3_2](https://user-images.githubusercontent.com/73010204/210160912-7f4692c1-e428-4a60-8bc0-b7d65a8546f2.png)</br>
![createS3_3](https://user-images.githubusercontent.com/73010204/210160916-b187ffba-72e3-4f55-8e0c-76ef1101f58a.png)</br>
![createS3_4](https://user-images.githubusercontent.com/73010204/210160919-b4033a2c-3146-43c6-9309-1f598ee138e3.png)</br>
Because we want to make this S3 bucket public acess so uncheck the _Block all public access_. We can do it by edit the _Permission_ later as well.</br>
Let's create the folder _images_to_classify_ and drag the .rec files created above to upload to, we will get </br>
![createS3_5](https://user-images.githubusercontent.com/73010204/210160966-42f92487-7335-43bd-a6b6-7b79148efffe.png)</br>
![createS3_6](https://user-images.githubusercontent.com/73010204/210161196-da7ce4c7-e431-477a-ad42-a50c9521a4c5.png)
 ### Make the S3 buket publicly accessible
 Check off _Block all public access_ is not enough to make the bucket public access, we need to edit its policy
![createS3_7](https://user-images.githubusercontent.com/73010204/210161348-f9217305-7c21-41e9-a212-faaed6d026d6.png)</br>
Replace your bucket name into the _"Resorce"_ tag, it should be like this
```sh
{
    "Version": "2008-10-17",
    "Statement": [
        {
            "Sid": "AllowPublicRead",
            "Effect": "Allow",
            "Principal": {
                "AWS": "*"
            },
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::nvirginia-lien-cats-dogs-buckets/*"
        }
    ]
}
```
![createS3_8](https://user-images.githubusercontent.com/73010204/210161350-1fd08311-b904-4edd-a5c3-f77236f2d7ad.png)</br>
Nice, and it becomes _public_. You can get the URL address and download .rec files from a browser.
![createS3_9](https://user-images.githubusercontent.com/73010204/210161352-0642f3d7-6d7d-4335-85fd-5b1fcdb7ba5b.png)</br>

## Create a SageMaker notebook instance
Choose SageMaker service, and create a new notebook instance. Because we created the bucket on N.Virginia region so it should be the same region here.</br>
![notebook1](https://user-images.githubusercontent.com/73010204/210161564-80c582f0-7914-40a2-8010-7acd0f9fd07e.png)</br>
![notebook2](https://user-images.githubusercontent.com/73010204/210161565-83d41ecd-5983-44c8-852c-85030e478b94.png)</br>
Enter your notebook instance name, choose _ml.t2.medium_ as the **instance type**, create new IAM role and leave other options as default</br>
![notebook3](https://user-images.githubusercontent.com/73010204/210161756-010cca44-9c49-4c22-995e-314b3fed3e51.png)</br>
Wait for a few minutes till the notebook instance become _in service_.</br>
Click on _juperlab_, create _new notebook_ with kernel as _conda python3_ and it's time to implement our AI model!

## Data preparation 
Download the data prepared above and transfer to S3 for use in training. 

```sh
! pip install --upgrade sagemaker
```

```sh
import sagemaker
from sagemaker import get_execution_role

role = get_execution_role()

sess = sagemaker.Session()
bucket = sess.default_bucket()
prefix = "my-catsdogs-fulltraining"
```

```sh
from sagemaker import image_uris

training_image = image_uris.retrieve(region=sess.boto_region_name, framework="image-classification")

print(training_image)
print(sess.boto_region_name)
```
You may get</br>
> 811284229777.dkr.ecr.us-east-1.amazonaws.com/image-classification:1
> us-east-1

```sh
import os
import urllib.request
import boto3

s3_train_key = "nvirginia-image-classification-full-training/train"
s3_validation_key = "nvirginia-image-classification-full-training/validation"
s3_train = "s3://{}/{}/".format(bucket, s3_train_key)
s3_validation = "s3://{}/{}/".format(bucket, s3_validation_key)

def download(url):
    filename = url.split("/")[-1]
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)

download("https://nvirginia-lien-cats-dogs-buckets.s3.amazonaws.com/images_to_classify/dataset_rec_train.rec")
download("https://nvirginia-lien-cats-dogs-buckets.s3.amazonaws.com/images_to_classify/dataset_rec_val.rec")

!aws s3 cp dataset_rec_train.rec $s3_train --quiet
!aws s3 cp dataset_rec_val.rec $s3_validation --quiet

print(s3_train)
print(s3_validation)
```
The result should be similar to</br>
> s3://sagemaker-us-east-1-597051996741/nvirginia-image-classification-full-training/train/</br>
> s3://sagemaker-us-east-1-597051996741/nvirginia-image-classification-full-training/validation/</br>
![notebook_s3](https://user-images.githubusercontent.com/73010204/210167464-cbf52abd-55f6-4927-8935-b2920b443bc8.png)</br>

## Training the model
Now that we are done with all the setup that is needed, we are ready to train our object detector.</br></br>

Training can be done by either calling SageMaker Training with a set of hyperparameters values to train with, or by leveraging SageMaker Automatic Model Tuning (AMT). AMT, also known as hyperparameter tuning (HPO), finds the best version of a model by running many training jobs on your dataset using the algorithm and ranges of hyperparameters that you specify. It then chooses the hyperparameter values that result in a model that performs the best, as measured by a metric that you choose.

We will try both methods are used for demonstration purposes, but the model that the HPO job creates will be the one that is used as the base one for incremental training. You can instead choose to use the model created by the standalone training job by changing the below variable _deploy_amt_model_ to False.
```sh
deploy_amt_model = True
```
### Training with SageMaker Training
To begin, let us create a _sageMaker.estimator.Estimator_ object to launch the training job.
#### Training parameters
There are two kinds of parameters that need to be set for training. The first one are the parameters for the training job. These include:
- **Training instance count**: This is the number of instances on which to run the training. When the number of instances is greater than one, then the image classification algorithm will run in distributed settings.
- **Training instance type**: This indicates the type of machine on which to run the training. We have to use GPU instances for image-classification training
- **Output path**: This the s3 folder in which the training output is stored
Apart from the above set of parameters, there are hyperparameters that are specific to the algorithm. These are:</br>
- **num_layers**: The number of layers (depth) for the network. We use 18 in this samples but other values such as 50, 152 can be used. Other than those numbers, you may get training error.
- **image_shape**: The input image dimensions,’num_channels, height, width’, for the network. It should be no larger than the actual image size. The number of channels should be same as the actual image.
- **num_classes**: This is the number of output classes for the new dataset. For cat and dog classification, we use 2.
- **num_training_samples**: This is the total number of training samples. It is set to 1000 for my data sample
- **mini_batch_size**: The number of training samples used for each mini batch.
- **epochs**: Number of training epochs.
- **learning_rate**: Learning rate for training.
- **top_k**: Report the top-k accuracy during training.

```sh
import sagemaker
s3_output_location = "s3://{}/{}/output".format(bucket, prefix)
ic = sagemaker.estimator.Estimator(
    training_image,
    role,
    instance_count=1,
    instance_type="ml.p2.xlarge",
    volume_size=50,
    max_run=360000,
    input_mode="File",
    output_path=s3_output_location,
    sagemaker_session=sess,
)
ic.set_hyperparameters(
    num_layers=18,
    image_shape="3,224,224",
    num_classes=2,
    num_training_samples=1000,
    mini_batch_size=64,
    epochs=5,
    learning_rate=0.01,
    top_k=2,
    precision_dtype="float32",
)
train_data = sagemaker.inputs.TrainingInput(
    s3_train,
    distribution="FullyReplicated",
    content_type="application/x-recordio",
    s3_data_type="S3Prefix",
)
validation_data = sagemaker.inputs.TrainingInput(
    s3_validation,
    distribution="FullyReplicated",
    content_type="application/x-recordio",
    s3_data_type="S3Prefix",
)

data_channels = {"train": train_data, "validation": validation_data}
ic.fit(inputs=data_channels, logs=True)
```
Here is my result. Because all I want is to demo how to image classification on Sagemaker, the training sample data has only 1000 images, that's why accuracy is not good.</br>
![1stTraing](https://user-images.githubusercontent.com/73010204/210168433-4e8edb98-6231-4b36-a92d-e29d8f95901c.png)</br>


# Reference
https://sagemaker-examples.readthedocs.io/en/latest/introduction_to_amazon_algorithms/imageclassification_caltech/Image-classification-incremental-training-highlevel.html

