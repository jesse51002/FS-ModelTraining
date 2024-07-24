import os
os.environ["CRYPTOGRAPHY_OPENSSL_NO_LEGACY"] = "1"

from sagemaker.pytorch import PyTorch
from sagemaker import get_execution_role, Session
import boto3


sagemaker_session = Session(boto3.session.Session(region_name='us-east-1'))
sagemaker_execution_role = get_execution_role(sagemaker_session=sagemaker_session)

pytorch_estimator = PyTorch(
    entry_point='train.py',
    source_dir=".",
    role=sagemaker_execution_role,
    instance_type='ml.g5.12xlarge',
    instance_count=1,
    framework_version='2.3.0',
    py_version='py311',
    max_run=431999,
    hyperparameters={
        "batch": 8,
        "iter": 65000,
        "arch": "swagan",
        "size": 1024,
        "distributed": None,
        "num_gpu": 4,
        "aws_checkpoint_name": "055000.pt",
        "upload_images_to_s3": None,
        "n_sample": 36,
        "lr_generator": 0.00005,
        "lr_discriminator": 0.00005,
        "discriminator_loss_limit": 0.7,
        "augment": None,
        "ada_length": 500000,
        "save_checkpoint_every": 5000,
        "ada_target": 0.57
    },
    metric_definitions=[
       {'Name': 'd_loss:error', 'Regex': 'd: (.*?);'},
       {'Name': 'g_loss:error', 'Regex': 'g: (.*?);'},
       {'Name': 'r1_val:error', 'Regex': 'r1: (.*?);'},
       {'Name': 'path_loss:error', 'Regex': 'path: (.*?);'},
       {'Name': 'mean_path_length_avg:error', 'Regex': 'mean_p: (.*?);'},
       {'Name': 'ada_aug_p:error', 'Regex': 'augment: (.*?);'},
    ],
    distribution={
        "torch_distributed": {
            "enabled": True,
            "parameters": {
                "nproc_per_node": 4
            }
        }
    }
)
# torchrun --nnodes 1 --nproc_per_node 4 train.py --arch swagan --batch 8 --distributed  --iter 800000 --num_gpu 4 --size 1024 --path data/accept_images_background_removed/ --augment --ada_length 100000 --upload_images_to_s3
# python train.py --arch swagan --batch 4 --iter 800000 --num_gpu 1 --size 1024 --aws_checkpoint_name 050000.pt --path data/accept_images_background_removed/ --n_sample 36 --augment --ada_length 100000 --discriminator_loss_limit 0.9
# ml.g5.12xlarge
pytorch_estimator.fit({'train': 's3://fs-upper-body-gan-dataset/accepted_images_background_removed/'})