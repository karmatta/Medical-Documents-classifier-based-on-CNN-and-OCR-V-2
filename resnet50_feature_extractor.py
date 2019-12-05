from keras.applications import ResNet50
from PIL import Image
import numpy as np

resnet50 = ResNet50(weights='imagenet', pooling=max, include_top = False) \

# Function to generate resnet features
def generate_image_features(image_path, reshape=(225, 225)):
    # GENERATING FEATURES
    image = Image.open(image_path)
    image.load()
    image = image.resize(reshape)
    x = np.asarray(self.image, dtype="int32" )
    x = np.expand_dims(self.x, axis=0) 
    img_features = resnet50.predict(x, verbose=0).squeeze().flatten()
    return img_features