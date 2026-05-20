import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


import tensorflow as tf
from tensorflow.python.client import device_lib

# Check if the GPU and CuDNN are recognized
print(device_lib.list_local_devices())
print(tf.test.is_built_with_cuda())
print(tf.config.list_physical_devices('GPU'))
