class CFG:
    data_url = "https://www.kaggle.com/api/v1/datasets/download/ghostbat101/lung-x-ray-image-clinical-text-dataset"
    dataset_path = "/content/dataset/Main dataset"
    img_size      = 256
    batch_size    = 32
    lr            = 3e-4
    epochs        = 15
    model_name    = "resnet50"
    num_classes   = 8
    class_names   = [
                    "Degenerative Infectious Diseases",
                    "Encapsulated Lesions",
                    "Obstructive Pulmonary Diseases",
                    "Higher Density",
                    "Mediastinal Changes",
                    "Normal",
                    "Lower Density",
                    "Chest Changes"
                    ]
    use_amp       = True

    def __init__(self, SKELETON_DIR):
      self.SKELETON_DIR   = SKELETON_DIR
      self.data_dir       = "dataset"
