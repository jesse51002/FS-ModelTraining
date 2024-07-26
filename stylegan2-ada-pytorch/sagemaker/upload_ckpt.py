import boto3
import os
import tarfile 


s3resource = boto3.client('s3')
BUCKET_NAME = "fs-upper-body-gan-dataset"

local_file = "checkpoints/network-snapshot-001800.pkl"
target_key = "training/model_run1_1800kimgs.pkl"


s3resource.upload_file(
        local_file, BUCKET_NAME, target_key)

