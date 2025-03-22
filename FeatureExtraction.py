import os
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input  
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img

from elpv_reader import load_dataset
paths, images, probs, types = load_dataset()

model = VGG16()
model = Model(inputs = model.inputs, outputs = model.layers[-2].output)
print(model.summary) 


