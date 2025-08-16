import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import pandas as pd
import numpy as np

class ImageDataset(Dataset):
    """
    PyTorch Dataset for images and pre-encoded labels stored in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with 'path' (image file paths) and 'label' (integer-encoded labels).

        img_size (int, optional): Target size to resize images (square). Default is 256.

        augment (bool, optional): Apply light augmentations if True. Default is False.

        path_col (str, optional): Column name for image paths in DataFrame. Default is "path".

        label_col (str, optional): Column name for labels in DataFrame. Default is "label".

    Returns:
        __getitem__ -> (torch.Tensor, torch.Tensor):
            - Image tensor of shape [3, img_size, img_size] normalized with ImageNet stats.
            - Integer label tensor (dtype=torch.long).
    """

    def __init__(self,
                 df,
                 img_size=256,
                 augment=False,
                 path_col="path",
                 label_col="label"):

        # Keep only the required columns and reset index for clean iteration
        self.df = df[[path_col, label_col]].reset_index(drop=True)

        # Store image paths as numpy array for fast access
        self.paths = self.df[path_col].values

        # Ensure labels are int64
        labels = self.df[label_col].values
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("Expected encoded integer labels in df[label_col].")
        self.labels = torch.as_tensor(labels, dtype=torch.long)

        # Base transforms
        base_tfms = [
            # Resize image to target size
            T.Resize((img_size, img_size)), 
            # Convert grayscale to 3-channel RGB         
            T.Grayscale(num_output_channels=3), 
            # Convert PIL image to tensor     
            T.ToTensor(), 
            # Normalize with ImageNet mean  and std.                       
            T.Normalize(mean=[0.485, 0.456, 0.406],   
                        std=[0.229, 0.224, 0.225])
        ]

        # Optional light augmentations if augment=True
        aug_tfms = [
            # Flip 50% of images horizontally
            T.RandomHorizontalFlip(p=0.5),
             # Small rotation to add variability           
            T.RandomRotation(degrees=7), 
            # Slight contrast adjustment           
            T.RandomAutocontrast(p=0.3)            
        ]

        # Compose transformations 
        self.transform = T.Compose((aug_tfms if augment else []) + base_tfms)

    def __len__(self):
        # Return total number of samples
        return len(self.paths)

    def __getitem__(self, index: int):
        """
        Load an image and its corresponding label.

        Args:
            index (int): Index of the sample.

        Returns:
            (torch.Tensor, torch.Tensor): (image_tensor, label_int)
        """
        # Get image path
        p = self.paths[index]

        # Load image from path
        with Image.open(p) as im:
            img = im.copy()  # Avoid issues with file pointer after closing

        # Apply transformations
        img = self.transform(img)

        # Get encoded label
        label = self.labels[index]

        return img, label
