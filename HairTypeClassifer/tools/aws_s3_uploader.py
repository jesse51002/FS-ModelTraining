import os

import boto3
from boto3.s3.transfer import TransferConfig
import shutil

s3resource = boto3.client('s3')
BUCKET_NAME = "fs-upper-body-gan-dataset"

def upload_aws_folder(abs_folder_path, rel_path):
    config = TransferConfig(multipart_threshold=1024*25, max_concurrency=10,
                        multipart_chunksize=1024*25, use_threads=True)
    
    abs_folder_path = abs_folder_path.replace("\\", "/")
    rel_path = rel_path.replace("\\", "/")
    
    zip_output_location = abs_folder_path + ".zip"
    
    print(f"making zip for {rel_path}")
    shutil.make_archive(abs_folder_path, 'zip', abs_folder_path)
    print(f"finished zipping for {rel_path}")
    
    print(f"uploading {rel_path}.zip")
    s3resource.upload_file(
        zip_output_location, BUCKET_NAME, rel_path + ".zip",
        ExtraArgs={ 'ACL': 'public-read', 'ContentType': 'video/mp4'},
        Config=config,
    )
    print(f"Finished uploading {rel_path}.zip")
    
    print(f"Deleting {rel_path} from local\n\n")
    os.remove(zip_output_location)
                
            
if __name__ == "__main__":
    chosen = -1
    while chosen < 1 or chosen > 2:
        print("""
        Choose upload mode
        1. Raw
        2. Clean
        """)
        chosen = int(input())

        if chosen >= 1 or chosen <= 2:
            print(f"""
            You have picked option {chosen}, are sure this action is irreversible\n
            Type 'confirm' to proceed
            """)

            if input() != "confirm":
                print(f"'confirm' was typed incorrectly. Restart...")
                chosen = -1
                continue
        else:
            print(f"{chosen} is an invalid choice, pick a valid choice")
            


