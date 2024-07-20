import os
os.environ["CRYPTOGRAPHY_OPENSSL_NO_LEGACY"] = "1"

import boto3
import tarfile


def download_checkpoints():
    bucket = "sagemaker-us-east-1-259645229668"

    train_jobs = [
        "pytorch-training-2024-07-18-21-47-07-445/",
        "pytorch-training-2024-07-18-03-06-42-963/",
        "pytorch-training-2024-07-17-07-24-50-696/",
    ]

    key_to_ckpts = "output/model.tar.gz"

    local_download_pth = "checkpoint/model.tar.gz"

    s3resource = boto3.client('s3')
    
    for job in train_jobs:
        key = job + key_to_ckpts
        print("Downloading:", key)
        s3resource.download_file(Bucket=bucket, Key=key, Filename=local_download_pth)
        
        with tarfile.open(local_download_pth) as f:
            f.extractall('checkpoint/')

        os.remove(local_download_pth)


if __name__ == "__main__":
    download_checkpoints()