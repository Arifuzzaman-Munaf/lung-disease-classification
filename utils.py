import os
import pandas as pd
from PIL import Image
import hashlib
import numpy as np
import matplotlib.pyplot as plt


def create_dataframe(path):
    """
    Reads images from subfolders, processes them 
    and returns a DataFrame with columns ['image', 'label'].

    arg
    path(str): Path to the main directory containing subfolders of images.

    returns:
    df: DataFrame with processed PIL Image objects and corresponding labels.
    """
    
    data = []

    # Loop through each subfolder in the main directory
    for subfolder_name in os.listdir(path):
        subfolder_path = os.path.join(path, subfolder_name)

        # Skip if not a directory
        if not os.path.isdir(subfolder_path):
            continue

        # Process each image in the subfolder
        for image_name in os.listdir(subfolder_path):
            image_path = os.path.join(subfolder_path, image_name)

            try:
                # Open image, resize to 256x256, and convert to grayscale
                image = Image.open(image_path).resize((256, 256)).convert('L')

                # Append image and label
                data.append((image, subfolder_name))

            except Exception as e:
                print(f"Error loading image {image_path}: {e}")

    # Create DataFrame with 'image' and 'label' columns
    return pd.DataFrame(data, columns=['image', 'label'])


def dataset_overview(df, image_col="image", label_col="label"):
    """
    Prints an overview of a dataset containing images and labels without altering the original DataFrame.

    arg:
    df : pd.DataFrame
    image_col : str, default="image"
        Name of the column containing image data.
    label_col : str, default="label"
        Name of the column containing label/category data.
    """

    # --- Helper: convert different image formats to bytes for hashing ---
    def to_bytes(img):
        """Convert an image (PIL, NumPy array, or bytes) into raw bytes."""
        if isinstance(img, bytes):
            return img
        if isinstance(img, Image.Image):
            return img.tobytes()
        if isinstance(img, np.ndarray):
            return img.tobytes()

        # Fallback: attempt conversion from array-like to PIL, then to bytes
        try:
            return Image.fromarray(np.array(img)).tobytes()
        except Exception:
            return str(img).encode("utf-8")  # Last resort: encode as text

    # Create a temporary copy for safe summarization
    tmp = df[[image_col, label_col]].copy()

    # Generate a short MD5 hex digest for each image
    # to avoids long unreadable bytes
    tmp["image_id"] = tmp[image_col].apply(lambda im: hashlib.md5(to_bytes(im)).hexdigest())

    # Compute dataset statistics
    num_rows = len(tmp)                          # Total number of rows
    unique_counts = tmp[["image_id", label_col]].nunique()  # Unique counts per column
    label_counts = tmp[label_col].value_counts()            # Distribution of labels
    summary_df = tmp[["image_id", label_col]].describe(include="all")  # Descriptive stats

    # Display results
    print(f"Number of rows: {num_rows}\n")
    print("Unique values per column:")
    print(unique_counts, "\n")
    print("Label distribution:")
    print(label_counts, "\n")
    print("Full DataFrame summary:")
    print(summary_df)

  
def show_images(df):
  """
  Shows one image from each class of the dataframe

  arg:
  df: the dataframe containig all images
  """
  # Get unique labels
  unique_labels = df['label'].unique()

  # Create a figure and axes for the subplots
  fig, axes = plt.subplots(1, len(unique_labels), figsize=(20, 5))

  # Iterate through unique labels and display one image for each
  for i, label in enumerate(unique_labels):
      # Get the first image for the current label
      image_to_display = df[df['label'] == label]['image'].iloc[0]

      # Display the image
      axes[i].imshow(image_to_display, cmap='gray')
      axes[i].set_title(label)
      axes[i].axis('off') # Hide axes

  plt.tight_layout()
  plt.show()
