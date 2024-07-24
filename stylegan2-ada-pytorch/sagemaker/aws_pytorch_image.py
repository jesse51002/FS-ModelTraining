from sagemaker import image_uris
print(image_uris.retrieve(framework='pytorch', region='us-east-1', version='1.8.0', py_version='py36', image_scope='training', instance_type='ml.g5.48xlarge'))