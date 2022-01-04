#!/usr/bin/python3

"""
[Depricated] Auxiliary script for splitting datasets.
"""

import os
import shutil
import PIL
import os
from PIL import Image

os.makedirs('images/train/class/', exist_ok=True) # 40,000 images
os.makedirs('images/val/class/', exist_ok=True)   #  1,000 images


def splitData():
    for root, directories, files in os.walk("dummyData/testing/ColorfulOriginal"):
        for i, file in enumerate(files):
            if(i % 10 == 0):
                outpathCol = root.replace("testing", "training")
                if(not os.path.exists(outpathCol)):
                    os.makedirs(outpathCol)

                src = os.path.join(os.getcwd(), root, file)
                trgt = os.path.join(os.getcwd(), outpathCol, file)
                shutil.move(src, trgt)

                grayPath = root.replace("ColorfulOriginal", "Gray")
                outpathGray = grayPath.replace("testing", "training")
                if(not os.path.exists(outpathGray)):
                    os.makedirs(outpathGray)

                src = os.path.join(os.getcwd(), grayPath, file)
                trgt = os.path.join(os.getcwd(), outpathGray, file)
                shutil.move(src, trgt)

def reshape():
    f = r'/home/ernests/Documents/Personal/universityNotes/MachineLearning/MachineLearningMiniProject/dummyData/ColorfulOriginal'
    for root, directories, files in os.walk(f):
        for file in files:
            f_img = root+"/"+file
            img = Image.open(f_img)
            img = img.resize((256,256))
            img.save(f_img)
            print(f"Resized {file}")

if __name__ == '__main__':
    reshape()
