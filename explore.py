import matplotlib.pyplot as plt
import hashlib
import numpy as np
import pandas as pd
from PIL import Image
from config import CFG as cfg

def plot_label_distribution(labels, class_names,split=""):
    """
    plot data distribution based on labels

    arg:
    label: list or series
          represents dataset labels in label encoded format
    class_names: list or series
                represents class names found in original dataset
    split: string; default=''
          represents data partition
    """
    # Count the number of samples for each class
    counts = [ sum(labels == c) for c in range(len(class_names)) ]
    
    # Plot a histogram of the distribution
    plt.title(f"{split if split else 'Data'} set distribution")
    plt.bar(class_names, counts)
    plt.xlabel('Class')
    plt.ylabel('Num examples')
    plt.xticks(rotation=90)
    plt.show()
 

def check_df_imbalance(labels, class_names, split='Data'):
    """
    check if dataset has imbalanced classes with classwise proportion

    arg:
    label: list or series
          represents dataset labels in label encoded format
    class_names: list or series
                represents class names found in original dataset
    split: string; default=''
          represents data partition
    """
    # Print header showing which dataset split's distribution is being analyzed
    print(f"{split if split else 'Data'} Set distribution in percentage:")

    # Count the number of samples for each class
    counts = np.array([sum(labels == c) for c in range(len(class_names))])

    # Convert counts to percentages
    props = (counts / labels.size) * 100

    # Print percentage distribution for each class
    for cls, prop in zip(class_names, props):
        print(f"{cls:>5} --> {prop:.2f}%")

    # Calculate imbalance ratio (max percentage / min percentage)
    ratio = max(props) / min(props)

    # Interpret and print imbalance severity
    if ratio > 3:
        print(f"Severely imbalanced {split} set with Min/Max ratio : {ratio:.1f}")
    elif ratio > 1.5:
        print(f"Moderate imbalanced {split} set with Min/Max ratio : {ratio:.1f}")
    elif ratio == 1:
        print(f"The {split} set properly balanced")
    else:
        print(f"The {split} set almost balanced")



def compute_duplicate(df, path_col="path", label_col="label"):
    """
    compute duplicate statistics for a path-based dataframe without altering df.

    args:
    df: pd.DataFrame 
      dataframe with 'path_col' (file paths) and 'label_col'.

    path_col: str, default="path"
       column containing image file paths.

    label_col (str, default="label"): column containing class labels.
    """
    # flag duplicates by repeated file paths 
    is_dup = df[path_col].duplicated(keep='first')

    # count duplicate rows excluding the first occurrence of each path
    duplicate_rows = int(is_dup.sum())

    # number of duplicate base on unique paths that appear >= 2 times
    duplicate_groups = int(df[path_col].value_counts().gt(1).sum())

    # class-wise duplicate counts
    if label_col in df.columns:
        classwise_dup = df.loc[is_dup, label_col].value_counts().sort_index()
    else:
        classwise_dup = pd.Series(dtype=int)

    stats = {
        "total_rows": len(df),
        "duplicate_rows": duplicate_rows,
        "duplicate_groups": duplicate_groups,
        "classwise_duplicate_rows": classwise_dup
    }

    # print summary
    print(f"Total rows: {stats['total_rows']}")
    print(f"Duplicate rows (excluding first occurrences): {stats['duplicate_rows']}")
    print(f"Duplicate groups : {stats['duplicate_groups']}")
    print("\nClass-wise duplicate rows:")
    if isinstance(classwise_dup, pd.Series) and not classwise_dup.empty:
        print(classwise_dup)
    else:
        print("No  duplicates found.")


def pixel_stats_overall(df, path_col="path", as_gray=True, sample=None, verbose=3):
    """
    Computes overall pixel intensity statistics across the dataset.

    args:
    df (pd.DataFrame): DataFrame with file paths.
    path_col (str, default="path"): Column with image file paths.
    as_gray (bool, default=True): Convert images to grayscale ('L') before stats.
    sample (int or None): If set, randomly sample N rows for speed.
    verbose (int, default=3): Print up to N open/convert errors for debugging.
    """

    # If sample is set, randomly select subset for faster computation
    data = df.sample(sample, random_state=42) if sample else df

    # Initialize accumulators for pixel stats
    count = 0         # Total number of pixels processed
    s1 = 0.0          # Sum of pixel values
    s2 = 0.0          # Sum of squared pixel values 
    gmin = float("inf")   # Global minimum pixel intensity
    gmax = float("-inf")  # Global maximum pixel intensity
    err = 0           # Counter for files that failed to load

    # Iterate over image paths
    for p in data[path_col].values:
        try:
            # Open image file
            with Image.open(p) as im:
                # Convert to grayscale if requested
                if as_gray and im.mode != "L":
                    im = im.convert("L")
                # Convert to NumPy array for numeric operations
                arr = np.asarray(im, dtype=np.float64)
        except Exception as e:
            # Handle failed file loads 
            if err < verbose:
                print(f"[skip] {p} -> {e}")
            err += 1
            continue

        # Update global min and max pixel values
        gmin = min(gmin, float(arr.min()))
        gmax = max(gmax, float(arr.max()))

        # Accumulate counts and sums for mean and std calculations
        count += arr.size
        s1 += arr.sum()
        s2 += (arr ** 2).sum()

    # If no pixels were processed, warn and exit
    if count == 0:
        print("No readable pixels found.")
        return {}

    # Compute mean pixel intensity
    mean = s1 / count

    # Compute variance and standard deviation
    var = (s2 / count) - (mean ** 2)
    std = float((var if var > 0 else 0) ** 0.5)

    # Prepare results dictionary
    stats = {
        "count": int(count),
        "min": float(gmin),
        "max": float(gmax),
        "mean": float(mean),
        "std": std
    }

    # Print results in readable format
    print("Overall Pixel Intensity Statistics",
          "(grayscale)" if as_gray else "(native modes)")
    for k, v in stats.items():
        print(f"{k:>6}: {v}")



def pixel_stats_by_class(df, path_col="path", label_col="label",
                         as_gray=True, sample_per_class=None, verbose=2):
    """
    Computes pixel intensity statistics for each class in the dataset and prints a summary table.

    args:
        df (pd.DataFrame): DataFrame with file paths.
        path_col (str, default="path"): Column with image file paths.
        as_gray (bool, default=True): Convert images to grayscale ('L') before stats.
        sample_per_class (int or None): If set, limit the number of images per class for analysis.
        verbose (int): Maximum number of error messages to print per class.
    """

    rows = []  # Store per-class statistics

    # Iterate over each unique class in sorted order
    for cls in sorted(df[label_col].unique()):
        sub = df[df[label_col] == cls]  # Filter rows belonging to this class

        # If sampling is enabled, randomly select a subset of images from the class
        if sample_per_class and len(sub) > sample_per_class:
            sub = sub.sample(sample_per_class, random_state=42)

        # Initialize accumulators for this class
        count = 0      # Total number of pixels processed
        s1 = 0.0       # Sum of pixel values
        s2 = 0.0       # Sum of squared pixel values 
        cmin = float("inf")   # Class min pixel intensity
        cmax = float("-inf")  # Class max pixel intensity
        err = 0        # Number of conversion errors for this class

        # Process each image in this class
        for p in sub[path_col].values:
            try:
                with Image.open(p) as im:
                    # Convert to grayscale if requested
                    if as_gray and im.mode != "L":
                        im = im.convert("L")
                    # Convert image to NumPy array for pixel-level calculations
                    arr = np.asarray(im, dtype=np.float64)
            except Exception as e:
                # Print errors for the first 'verbose' failed images
                if err < verbose:
                    print(f"[{cls}] skip {p} -> {e}")
                err += 1
                continue

            # Update per-class pixel intensity statistics
            cmin = min(cmin, float(arr.min()))
            cmax = max(cmax, float(arr.max()))
            count += arr.size
            s1 += arr.sum()
            s2 += (arr ** 2).sum()

        # If no pixels were processed for this class, append NaNs
        if count == 0:
            rows.append([cls, 0, float("nan"), float("nan"),
                         float("nan"), float("nan")])
        else:
            # Compute mean and standard deviation for this class
            mean = s1 / count
            var = (s2 / count) - (mean ** 2)
            std = float((var if var > 0 else 0) ** 0.5)
            rows.append([cls, int(count), float(cmin), float(cmax),
                         float(mean), std])

    # Convert results to DataFrame for pretty printing
    out = pd.DataFrame(rows, columns=["class", "count", "min", "max", "mean", "std"])
    print(out)  # Display the table

    # Plot mean pixel intensity per class
    plt.figure(figsize=(8, 5))
    plt.bar(out["class"].astype(str), out["mean"], color="skyblue", edgecolor="black")
    plt.xlabel("Class")
    plt.ylabel("Mean Pixel Intensity")
    plt.title("Mean Pixel Intensity by Class")
    plt.show()

def plot_pixel_histogram(df, path_col="path", as_gray=True, bins=256, sample=300, verbose=3):
    """
    Plots a pixel intensity histogram from a random subset of images.

    args:
    df (pd.DataFrame): DataFrame with file paths.

    path_col (str, default="path"): Column with image file paths.

    bins(int, default=256): Number of bins in the histogram (256 for full 8-bit pixel range).

    as_gray (bool, default=True): Convert images to grayscale ('L') before stats.

    sample(int, default=300): Number of random images to sample from the dataset for plotting.

    verbose (int): Maximum number of error messages to print per class.

    """

    # Randomly sample subset of images (for speed) if sample limit is set
    sub = df.sample(min(sample, len(df)), random_state=42) if sample else df

    chunks = []  # Will hold flattened pixel intensity arrays
    err = 0      # Count of files that failed to open

    # Iterate over sampled image paths
    for p in sub[path_col].values:
        try:
            # Open image file
            with Image.open(p) as im:
                # Convert to grayscale if requested
                if as_gray and im.mode != "L":
                    im = im.convert("L")
                # Convert to NumPy array
                arr = np.asarray(im)
        except Exception as e:
            # If an error occurs and we haven't printed too many yet, show it
            if err < verbose:
                print(f"[skip] {p} -> {e}")
            err += 1
            continue

        # Flatten image array into 1D and store
        chunks.append(arr.ravel())

    # If no valid images were processed, exit
    if not chunks:
        print("no readable pixels to plot.")
        return

    # Concatenate all pixel arrays into one vector
    pix = np.concatenate(chunks)

    # Create histogram plot
    plt.figure(figsize=(6,4))
    plt.hist(pix, bins=bins)
    plt.title("Pixel Intensity Histogram (sampled)")
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
