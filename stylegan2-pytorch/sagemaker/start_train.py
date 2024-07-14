from sagemaker.pytorch import PyTorch


pytorch_estimator = PyTorch(
    'pytorch-train.py',
    instance_type='ml.p3.2xlarge',
    instance_count=1,
    framework_version='1.8.0',
    py_version='py3',
    hyperparameters = {
        "batch": 16,
        "iter": 800000,
        "arch": "swagan",
        "size": 1024,
    },
    metric_definitions=[
       {'Name': 'd_loss:error', 'Regex': 'd: (.*?);'},
       {'Name': 'g_loss:error', 'Regex': 'g: (.*?);'},
       {'Name': 'r1_val:error', 'Regex': 'r1: (.*?);'},
       {'Name': 'path_loss:error', 'Regex': 'path: (.*?);'},
       {'Name': 'mean_path_length_avg:error', 'Regex': 'mean path: : (.*?);'},
       {'Name': 'ada_aug_p:error', 'Regex': 'augment: (.*?);'},
    ],
)


pytorch_estimator.fit({'train': 's3://fs-upper-body-gan-dataset/accepted_images_background_removed/',})