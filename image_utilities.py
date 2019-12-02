# Python script to for image utilitie functions
# Author: Karthik D

from PIL import Image
import numpy as np

# Make image tensors
def make_image_tensors(image_path):
    image = Image.open(image_path)
    image = image.resize((225, 225))
    image.load()
    data = np.asarray( image, dtype="int32" )
    return data