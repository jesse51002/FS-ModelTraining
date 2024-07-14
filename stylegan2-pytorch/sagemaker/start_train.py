import os
os.environ["CRYPTOGRAPHY_OPENSSL_NO_LEGACY"] = "1"

from sagemaker.pytorch import PyTorch
from sagemaker import get_execution_role, Session
import boto3


sagemaker_session = Session(boto3.session.Session(region_name='us-east-1'))
sagemaker_execution_role = get_execution_role(sagemaker_session=sagemaker_session)

pytorch_estimator = PyTorch(
    'train.py',
    role=sagemaker_execution_role,
    image_uri="259645229668.dkr.ecr.us-east-1.amazonaws.com/stylegan:1.0",
    instance_type='ml.g4dn.xlarge',
    instance_count=1,
    framework_version='1.8.0',
    py_version='py3',
    hyperparameters={
        "batch": 8,
        "iter": 800000,
        "arch": "swagan",
        "size": 1024,
        "sagemaker": None
    },
    metric_definitions=[
       {'Name': 'd_loss:error', 'Regex': 'd: (.*?);'},
       {'Name': 'g_loss:error', 'Regex': 'g: (.*?);'},
       {'Name': 'r1_val:error', 'Regex': 'r1: (.*?);'},
       {'Name': 'path_loss:error', 'Regex': 'path: (.*?);'},
       {'Name': 'mean_path_length_avg:error', 'Regex': 'mean path: : (.*?);'},
       {'Name': 'ada_aug_p:error', 'Regex': 'augment: (.*?);'},
    ],
)


pytorch_estimator.fit({'train': 's3://fs-upper-body-gan-dataset/accepted_images_background_removed/'})