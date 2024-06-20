import os

import torch
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
import numpy as np

from models.FaceDetector import output_bb
from models.FacerParsing import FacerModel, FacerDetection, FACER_CLASSES
from models.p3m_matting.inference import remove_background
from models.face_parsing import get_segmentation, IBUG_CLASSES


data_folder = "./data"
hair_only_folder = os.path.join(data_folder, "hair_only")
full_img_folder = os.path.join(data_folder, "full_img")

BATCH_SIZE = 4
BACKGROUND_COLOR = 255


def preprocess_dataset():
    """
    Preprocesses the dataset by generating hair-only images for each hair type.
    """
    # Initialize the facer model with a face detector
    facer = FacerModel(face_detector=FacerDetection(), device="cuda")
    
    # Iterate over each hair type
    for hair_type in os.listdir(full_img_folder):
        hair_type_path = os.path.join(full_img_folder, hair_type)
        
        # Get a list of image paths for the current hair type
        imgs = [os.path.join(hair_type_path, img) for img in os.listdir(hair_type_path)]
        
        # Create a batched input tensor
        batched_input = torch.zeros((len(imgs), 3, 1024, 1024))
        
        # Process each image in the batch
        for i, img_path in enumerate(imgs):
            image = Image.open(img_path)
            np_image = np.array(image.convert('RGB'))
            torch_img = torch.tensor(np_image).permute(2, 0, 1)
            torch_img = F.interpolate(torch_img.unsqueeze(0), size=(1024, 1024), mode='nearest').squeeze()
            batched_input[i] = torch_img

        # Initialize tensors for storing segmentation, ibug set, and human segmentation masks
        human_segs = torch.zeros((0, 1024, 1024)).cuda()
        seg_targets = torch.zeros((0, 1024, 1024)).cuda()
        ibug_sets = torch.zeros((0, 1024, 1024)).cuda()
    
        # Process sub-batches of images
        for i in range(0, len(imgs), BATCH_SIZE):
            cur_batched_input = batched_input[i: min(i+BATCH_SIZE, len(imgs))]

            # Get the human matting segmentation mask for the current batch
            _, cur_human_seg = remove_background(cur_batched_input)

            # Get the keypoints results for the current batch
            cur_ibug_sets = get_segmentation(cur_batched_input)

            # Get the face parsing results for the current batch
            cur_seg_targets = facer.inference(cur_batched_input)[0]
            
            human_segs = torch.cat([human_segs, cur_human_seg], dim=0)
            ibug_sets = torch.cat([ibug_sets, cur_ibug_sets], dim=0)
            seg_targets = torch.cat([seg_targets, cur_seg_targets], dim=0)
            
            print(f"{hair_type}: Model processing: {i} / {len(imgs)}")
            
            

        # Convert tensors to numpy arrays
        seg_targets = seg_targets.float().data.cpu().numpy()
        human_segs = human_segs.data.cpu().numpy()
        ibug_sets = ibug_sets.data.cpu().numpy()
        
        # Generate a boolean array indicating which pixels are hair
        hair_bool_array = np.where(
            (seg_targets == FACER_CLASSES["hair"]) & 
            (ibug_sets == IBUG_CLASSES["hair"]) &
            (human_segs > 0.5),
            1, 0
            )
        
        # Generate hair-only images
        batched_input = batched_input.detach().cpu().numpy()
        
        hair_only_ouputs = np.where(hair_bool_array == 1, batched_input, BACKGROUND_COLOR)

        # Transpose the input tensor
        batched_input = np.transpose(batched_input, (0, 2, 3, 1))
        
        # Save each hair-only image
        for i in range(hair_only_ouputs.shape[0]):
            file_name = os.path.basename(imgs[i])
            
            target_pth = os.path.join(hair_only_folder, hair_type, file_name)
            
            img = Image.fromarray(hair_only_ouputs[i])
            img.save(target_pth)
            
            