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
# create cats_dogs_dataset_rec_train.lst file
python C:/Users/liennt/AppData/Local/Programs/Python/Python36/Lib/site-packages/mxnet/tools/im2rec.py ./cats_dogs_dataset_rec_train ./train/ --recursive --list --num-thread 8

# create cats_dogs_dataset_rec_train.rec va cats_dogs_dataset_rec_train.idx
python C:/Users/liennt/AppData/Local/Programs/Python/Python36/Lib/site-packages/mxnet/tools/im2rec.py ./cats_dogs_dataset_rec_train ./train/ --recursive --pass-through --pack-label --num-thread 8
```
Do almost the same to create .lst and .rec file for test folder
```sh
# create cats_dogs_dataset_rec_val.lst file
python C:/Users/liennt/AppData/Local/Programs/Python/Python36/Lib/site-packages/mxnet/tools/im2rec.py ./cats_dogs_dataset_rec_val ./test/ --recursive --list --num-thread 8

# create cats_dogs_dataset_rec_val.lst file
python C:/Users/liennt/AppData/Local/Programs/Python/Python36/Lib/site-packages/mxnet/tools/im2rec.py ./cats_dogs_dataset_rec_val ./test --recursive --pass-through --pack-label --no-shuffle --num-thread 8
```
![image](https://user-images.githubusercontent.com/73010204/216799934-c248d0b3-69c8-41ef-ba70-ae1e71bd3ebf.png)</br>
Then your folder will look like</br>
![image](https://user-images.githubusercontent.com/73010204/216799941-ec27a52c-d614-492d-8512-95c08d50c1d1.png)</br>

## Upload to S3 bucket. Make public access
### Create a S3 bucket
Login your AWS account, go to S3 bucket service and Create a bucket</br>
The region of the bucket **must be same** with the region of the Sagemaker notebook instance you will create later. We don't need to specifiy where we are living as the region. Actually, I tried Osaka and then I got error of not support neither _ml.p3.2xlarge_ nor _ml.p2.xlarge_ instance_type when training.</br>
Fill your bucket name</br>
![image](https://user-images.githubusercontent.com/73010204/216795007-8e8a59da-f49e-4c56-aacf-b22c9ea404f0.png)</br>
Because we want to make this S3 bucket public acess so uncheck the _Block all public access_. We can do it by edit the _Permission_ later as well.</br>
![image](https://user-images.githubusercontent.com/73010204/216795035-1e5b92b6-e847-40d7-98ac-22f66e93eee9.png)</br>
![image](https://user-images.githubusercontent.com/73010204/216795066-87176ca0-cbd0-4cd0-8a33-7c9b1ba7f84a.png)</br>
Let's create the folder _images_to_classify_ and drag the .rec files created above to upload to, we will get </br>
![image](https://user-images.githubusercontent.com/73010204/216795176-847ba837-428c-474f-bd1f-8ee90d46de8c.png)</br>
↓
![image](https://user-images.githubusercontent.com/73010204/216795126-ed1238cb-a946-4f14-8634-6d1934ca0a51.png)</br>

![image](https://user-images.githubusercontent.com/73010204/216799979-8d5a0f57-b98b-4304-9d00-2fd7648d20d9.png)
 ### Make the S3 buket publicly accessible
 Check off _Block all public access_ is not enough to make the bucket public access, we need to edit its policy
![image](https://user-images.githubusercontent.com/73010204/216795275-3910665e-5d21-4ff6-863e-5450c658e21c.png)</br>
Replace your bucket name into the _"Resource"_ tag, it should be like this
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
            "Resource": "arn:aws:s3:::lien-cats-dogs-bucket/*"
        }
    ]
}
```
![image](https://user-images.githubusercontent.com/73010204/216795348-9c1d3c27-758a-4624-8203-a11c76fb7093.png)</br>
Nice, and it becomes _public_. You can get the URL address and download .rec files from a browser.
![image](https://user-images.githubusercontent.com/73010204/216795379-a3740860-bc7e-431c-a477-00f60eca9d26.png)</br>
Get .rec file url link:</br>
![image](https://user-images.githubusercontent.com/73010204/215301007-0631f885-ece3-4049-84ee-735200f1bee0.png)</br>
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

s3_train_key = "cats-dogs-classification-full-training/train"
s3_validation_key = "cats-dogs-classification-full-training/validation"
s3_train = "s3://{}/{}/".format(bucket, s3_train_key)
s3_validation = "s3://{}/{}/".format(bucket, s3_validation_key)

def download(url):
    filename = url.split("/")[-1]
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)
# download to this current instance memory
download("https://lien-cats-dogs-bucket.s3.amazonaws.com/images_to_classify/cats_dogs_dataset_rec_train.rec")
download("https://lien-cats-dogs-bucket.s3.amazonaws.com/images_to_classify/cats_dogs_dataset_rec_val.rec")
#copy to s3 bucket
!aws s3 cp cats_dogs_dataset_rec_train.rec $s3_train --quiet
!aws s3 cp cats_dogs_dataset_rec_val.rec $s3_validation --quiet

print(s3_train)
print(s3_validation)
```
The result should be similar to</br>
![image](https://user-images.githubusercontent.com/73010204/216800182-a0566d6a-2c59-4ca0-bff6-faf2d7ddfa8f.png)</br>

## Training the model
Now that we are done with all the setup that is needed, we are ready to train our object detector.</br></br>

Training can be done by either calling SageMaker Training with a set of hyperparameters values to train with, or by leveraging SageMaker Automatic Model Tuning (AMT). AMT, also known as hyperparameter tuning (HPO), finds the best version of a model by running many training jobs on your dataset using the algorithm and ranges of hyperparameters that you specify. It then chooses the hyperparameter values that result in a model that performs the best, as measured by a metric that you choose.

We will try both methods are used for demonstration purposes, but the model that the HPO job creates will be the one that is used as the base one for incremental training. You can instead choose to use the model created by the standalone training job by changing the below variable _deploy_amt_model_ to False.
```sh
deploy_amt_model = True
```
### Training with SageMaker Training
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
    instance_type="ml.p3.2xlarge",
    volume_size=50,
    max_run=360000,
    input_mode="File",
    output_path=s3_output_location,
    sagemaker_session=sess,
)
ic.set_hyperparameters(
    num_layers=18,
    use_pretrained_model = 1,
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
Here is my result. Because all I want is to demo how to image classification on Sagemaker, the training sample data has only 1000 images, we used transfer learning by setting _use_pretrained_model = 1,_.</br>
![image](https://user-images.githubusercontent.com/73010204/216800326-7760b6c3-1426-4aca-8a3a-fa90b6eed72c.png)</br>

### Training with Automatic Model Tuning (HPO)
As mentioned above, instead of manually configuring our hyper parameter values and training with SageMaker Training, we’ll use Amazon SageMaker Automatic Model Tuning.
```sh
import time
from sagemaker.tuner import IntegerParameter, ContinuousParameter
from sagemaker.tuner import HyperparameterTuner

job_name = "CAT-DOG-ic-" + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
print("Tuning job name: ", job_name)

# Image Classification tunable hyper parameters can be found here https://docs.aws.amazon.com/sagemaker/latest/dg/IC-tuning.html
hyperparameter_ranges = {
    "beta_1": ContinuousParameter(1e-6, 0.999, scaling_type="Auto"),
    "beta_2": ContinuousParameter(1e-6, 0.999, scaling_type="Auto"),
    "eps": ContinuousParameter(1e-8, 1.0, scaling_type="Auto"),
    "gamma": ContinuousParameter(1e-8, 0.999, scaling_type="Auto"),
    "learning_rate": ContinuousParameter(1e-6, 0.5, scaling_type="Auto"),
    "mini_batch_size": IntegerParameter(8, 64, scaling_type="Auto"),
    "momentum": ContinuousParameter(0.0, 0.999, scaling_type="Auto"),
    "weight_decay": ContinuousParameter(0.0, 0.999, scaling_type="Auto"),
}

# Increase the total number of training jobs run by AMT, for increased accuracy (and training time).
max_jobs = 6
# Change parallel training jobs run by AMT to reduce total training time, constrained by your account limits.
# if max_jobs=max_parallel_jobs then Bayesian search turns to Random.
max_parallel_jobs = 1


hp_tuner = HyperparameterTuner(
    ic,
    "validation:accuracy",
    hyperparameter_ranges,
    max_jobs=max_jobs,
    max_parallel_jobs=max_parallel_jobs,
    objective_type="Maximize",
)

# Launch a SageMaker Tuning job to search for the best hyperparameters
hp_tuner.fit(inputs=data_channels, job_name=job_name)
```
It may take few minutes to complete</br>
![image](https://user-images.githubusercontent.com/73010204/216801926-9e0c8b86-d92a-4c67-bb67-711ed3cbaff8.png)</br>
![image](https://user-images.githubusercontent.com/73010204/216801984-5ba06cd8-9835-4336-bcbc-d522e5ced5cd.png)</br>
### Prepare for incremental training
Incremental training is a training technique in machine learning where the model is trained on small chunks of data, rather than the entire dataset, at a time. This allows the model to continuously learn and adapt to new information as it becomes available, rather than having to retrain on the entire dataset every time there is a change. Incremental training can improve the training efficiency and reduce the computational resources required, especially when dealing with large datasets.</br>
Incremental training is useful in several situations:
- Large datasets: When the dataset is too large to fit into memory, incremental training can be used to train the model on smaller chunks of data at a time, which allows the model to make progress without having to wait for the entire dataset to be loaded.
- Data Streams: In cases where the data is constantly changing and streaming in real-time, incremental training can be used to update the model as new data becomes available. This can be useful in applications such as online advertising, fraud detection, and customer behavior analysis.
- Cost-effective: Training a model on a large dataset can be computationally expensive, and incremental training can reduce the cost by training on small chunks of data at a time.
- Adaptation to new data: When the distribution of the data changes over time, incremental training can be used to adapt the model to these changes and improve its performance.

In summary, incremental training is useful when the data is large, constantly changing, computationally expensive, or when the model needs to adapt to changes in the data distribution.</br>
It may be unnecessary in this sample but just do it. :D
Or you can apply the _hp_tuner_ to deply right away.

```sh
# Print the location of the model data from the tuning job's best training (or the previous standlone training)
model_data = (hp_tuner.best_estimator() if deploy_amt_model else ic).model_data
print("***1.deploy_amt_model: ", deploy_amt_model)
print("***2.hp_tuner.best_estimator().model_data: ", hp_tuner.best_estimator().model_data)
print("***3.ic.model_data: ",ic.model_data)
print("***4.model_data: ",model_data)
# Prepare model channel in addition to train and validation
model_data_channel = sagemaker.inputs.TrainingInput(
    model_data,
    distribution="FullyReplicated",
    s3_data_type="S3Prefix",
    content_type="application/x-sagemaker-model",
)

data_channels = {"train": train_data, "validation": validation_data, "model": model_data_channel}
```
![image](https://user-images.githubusercontent.com/73010204/216802004-47b12901-730c-4641-89c0-0d127492fd97.png)</br>
### Start another training
We do training again but with the best tuning parameter found. The number of classes, input image shape and number of layers should be the same as the previous training since we are starting with the same model. Other parameters, such as learning_rate, mini_batch_size, etc., can vary.
```sh
incr_ic = sagemaker.estimator.Estimator(
    training_image,
    role,
    instance_count=1,
    instance_type="ml.p3.2xlarge",
    volume_size=50,
    max_run=360000,
    input_mode="File",
    output_path=s3_output_location,
    sagemaker_session=sess,
)
incr_ic.set_hyperparameters(
    num_layers=18,
    use_pretrained_model = 1,
    image_shape="3,224,224",
    num_classes=2,
    num_training_samples=1000,
    mini_batch_size=64,
    epochs=5,
    learning_rate=0.01,
    top_k=2,
)

incr_ic.fit(inputs=data_channels, logs=True)
```
As you can see from the logs, the training starts with the previous model and hence the accuracy for the first epoch itself is higher.</br>
![image](https://user-images.githubusercontent.com/73010204/216802756-6e6e49cf-659c-4202-b5da-7e0798f29bce.png)</br>

## Realtime inference
### Download test image
We need to pre-processing for input image before testing it
```sh
from PIL import Image

NORMALIZED_WID = 224
NORMALIZED_HEI = 224

def calculate_image_crop_box_by_center(image):
    # box=(left, upper, right, lower)
    width, height = image.size # Output: (499, 375)
    print(f"in: {image.size}")
    if (width == height) and (height == NORMALIZED_HEI):
        return(0, 0 , NORMALIZED_WID, NORMALIZED_WID)
    center_x = width/2
    center_y = height/2
    if (width <= height):
        top = center_y - width/2
        left = 0
        return (left, top, left + width, top + width)
    else:
        top = 0
        left = center_x - height/2
        return (left, top, left + height, top + height)

def resize_scale_image_by_box(image, box):
    outImage = image.crop(box)
    outImage.thumbnail((NORMALIZED_WID, NORMALIZED_HEI))
    print(f"out: {outImage.size}")
    return outImage


image_url = "https://lien-cats-dogs-bucket.s3.amazonaws.com/test/Cute_dog.jpg"
download(image_url)

file_name = image_url.split("/")[-1]
image = Image.open(file_name)
print(file_name)

image.show()

print("image pre-processing..")
croppedBox = calculate_image_crop_box_by_center(image)
image = resize_scale_image_by_box(image, croppedBox)
image.show()
normalized_file_name = "normalized_cats_dogs.jpg"
image.save(normalized_file_name)
print("image pre-processing completed")
```
![image](https://user-images.githubusercontent.com/73010204/216803232-01d1a80e-33ec-4eae-bb60-4323722e9232.png)</br>
### Deploy
### Clean up

# Reference
https://sagemaker-examples.readthedocs.io/en/latest/introduction_to_amazon_algorithms/imageclassification_caltech/Image-classification-incremental-training-highlevel.html</br>

https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/imageclassification_caltech/Image-classification-fulltraining.ipynb

