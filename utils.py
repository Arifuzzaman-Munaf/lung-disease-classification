import os
import pandas as pd
from PIL import Image
import hashlib
import re
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split

def create_dataframe(path, valid_exts=(".jpg",), max_workers=8):
    """
    returns a DataFrame with ['path', 'label'] only.

    args:
    path (str): Root directory containing subfolders (each subfolder is a label).

    valid_exts (tuple): Allowed image extensions.

    max_workers (int): Number of threads for parallel directory scanning.

    returns:
    pd.DataFrame: DataFrame with 'path' (image file path) and 'label' (class name).
    """
    
    # Collect candidate file paths per class 
    #parallel I/O scan
    def _list_files(class_dir):
        try:
            return [
                os.path.join(class_dir, f)
                for f in os.listdir(class_dir)
                if f.lower().endswith(valid_exts)
            ]
        except Exception:
            return []

    # Get subfolder names (class labels)
    labels = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    records = []

    # Parallel scan of each class folder
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_list_files, os.path.join(path, lbl)): lbl for lbl in labels}
        for fut in as_completed(futures):
            lbl = futures[fut]
            for p in fut.result():
                records.append((p, lbl))

    # return the structured dataframe
    return pd.DataFrame(records, columns=["path", "label"])



def dataset_overview(df, path_col = "path", label_col = "label"):
    """
    Print an overview of a dataset where images are referenced by file paths.

    Args:
        df (pd.DataFrame): DataFrame containing 'path' and 'label' columns.
        path_col (str, optional): Column name containing image file paths. Defaults to "path".
        label_col (str, optional): Column name containing labels. Defaults to "label".

    """
    # compute MD5 for a file without loading it into memory all at once
    def file_md5(path: str, chunk_size: int = 1 << 20) -> str:
        """Return MD5 hex digest for a file, reading in chunks to limit memory use."""
        md5 = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                md5.update(chunk)
        return md5.hexdigest()

    # Work on a shallow copy to avoid any accidental mutation
    tmp = df[[path_col, label_col]].copy()

    # Compute a short, readable fingerprint per image file
    tmp["image_id"] = tmp[path_col].apply(file_md5)

    # Core stats
    num_rows = len(tmp)
    unique_counts = tmp[["image_id", label_col]].nunique()
    label_counts = tmp[label_col].value_counts()
    summary_df = tmp[["image_id", label_col]].describe(include="all")

    # Display results
    print(f"Number of rows: {num_rows}\n")
    print("Unique values per column:")
    print(unique_counts, "\n")
    print("Label distribution:")
    print(label_counts, "\n")
    print("Full DataFrame summary:")
    print(summary_df)


def show_images(df, path_col = "path", label_col = "label"):
    """
    Display one example image from each class using file paths in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing image paths and labels.
        path_col (str, optional): Column name containing image file paths. Defaults to "path".
        label_col (str, optional): Column name containing labels. Defaults to "label".

    Notes:
        - If a class has no readable image (missing/corrupt), it is skipped with a warning.
        - Uses grayscale colormap for single-channel images; RGB otherwise.
    """
    # Collect unique labels (classes)
    unique_labels = list(df[label_col].unique())
    if len(unique_labels) == 0:
        print("No labels found.")
        return

    # handle the single-class case so axes is always iterable
    n_classes = len(unique_labels)
    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 5))
    if n_classes == 1:
        axes = [axes]  # make it iterable

    # Iterate classes and show the first readable image for each class
    shown = 0
    for ax, label in zip(axes, unique_labels):
        subset = df[df[label_col] == label]
        img_shown = False

        for path in subset[path_col]:
            try:
                with Image.open(path) as im:
                    # Decide how to display: grayscale for 'L' or single-channel
                    if im.mode == "L":
                        ax.imshow(im, cmap="gray")
                    else:
                        ax.imshow(im)
                    ax.set_title(str(label))
                    ax.axis("off")
                    img_shown = True
                    shown += 1
                    break
            except Exception as e:
                # If a given file fails, try the next one in the same class
                print(f"Warning: could not open {path} ({e}). Trying next sample...")

        if not img_shown:
            ax.set_title(f"{label} (no readable image)")
            ax.axis("off")

    # If there were more axes than images shown , hide extra axes
    for ax in axes[shown:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def split_dataset(df, test_size=0.2, val_size=0.3):
    train_val_df, test_df = train_test_split(df, test_size=test_size, stratify=df['label'])
    train_df, val_df = train_test_split(train_val_df, test_size=val_size, stratify=train_val_df['label'])
    
    return train_df, val_df, test_df



def create_image_dataframe(df, col="path", width=4):
    """
    Create a copy of df where:
      - 'Merged Data (image+text)' is replaced with 'Main dataset'
      - image numbers are zero-padded (e.g., image_140 -> image_0140)

    Args:
        df (pd.DataFrame): input dataframe
        col (str): column containing file paths
        width (int): width for zero-padding (default=4)

    Returns:
        pd.DataFrame: transformed copy of df
    """
    def transform_path(path):
        # Replace folder name
        path = path.replace("Merged Data (image+text)", "Main dataset")
        # Zero-pad image numbers
        path = re.sub(r"(image_)(\d+)",
                      lambda m: f"{m.group(1)}{int(m.group(2)):0{width}d}",
                      path)
        return path

    df_image = df.copy()
    df_image[col] = df_image[col].apply(transform_path)
    return df_image
