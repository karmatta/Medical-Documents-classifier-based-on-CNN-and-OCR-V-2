# Python script to define run OCR on images
# Author: Karthik D

import pyocr
import pyocr.builders
from PIL import Image
import re
import os, glob
import pandas as pd
import numpy as np

# Function to preprocess text
def preprocess_tokens(text):
    text = re.sub('[0-9\.=\\|\-;/,%$#@!&`:+\*\(\)\[\]\{\}\'\\n\"]', ' ', text)
    text = ' '.join(text.split())
    return text

# OCR Wrpper function
def get_text_from_image(image):
    tools = pyocr.tesseract
    #text = tools.image_to_string(Image.open(image))
    text = tools.image_to_string(image)
    text = preprocess_tokens(text)
    return text

# Function to create a DF from raw images
def create_data_df(data_path):
    file_list = []
    for folder in glob.iglob(os.path.join(data_path, "*")):
        dirname = os.path.basename(folder)
        print("Processing: ", dirname)
        for image_path in glob.iglob(os.path.join(folder, "*")):
            if os.path.isfile(image_path):
                # Extract filename
                filename = os.path.splitext(os.path.basename(image_path))[0]
                try:
                    # Get text from image using OCR
                    text = get_text_from_image(image_path)
                    # Append every file to list
                    file_list.append([filename, image_path, text, dirname])
                except Exception as e:
                    print(e)
    lookup_df = pd.DataFrame(file_list, columns = ['filename', 'path', "x_txt", 'y'])
    return lookup_df
