class CFG:
    SKELETON_DIR  = '/content/drive/MyDrive/DLE602'
    data_dir      = f"{SKELETON_DIR}/data"
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
