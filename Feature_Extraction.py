# Loading Required Libraries

from matplotlib import image
from matplotlib import pyplot
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import seaborn as sns
from gtda.homology import CubicalPersistence
from gtda.diagrams import BettiCurve
import glob

# Setting Homology Dimension


homology_dimensions = [1]
CP = CubicalPersistence(
    homology_dimensions=homology_dimensions,
    coeff=3,
    n_jobs=1
)
BC = BettiCurve()

# Reading images from file

file = "C://Users/16823/Desktop/Retinal_Papers/APTOS Dataset/train_images/"

img_file = list(glob.glob1(file, "*.jpeg"))
img = []
for i in img_file:
    img.append(i)

# Feature Extraction for Grayscale images

data = []
for i in img:
    image_path = file + i
    gray_h1 = Image.open(image_path).convert('L')
    im_gray_h1 = np.array(gray_h1)
    diagram_h1_0 = CP.fit_transform(np.array(im_gray_h1)[None, :, :])
    y_betti_curves_h1_0 = BC.fit_transform(diagram_h1_0)
    data.append(np.reshape(y_betti_curves_h1_0, 100))
df0 = pd.DataFrame(data)
#df0["label"] = [0] * len(data)

df0.to_excel("df_train_n_Betti_1.xlsx")


# Feature Extraction for RGB channel
data_RGB = []
for i in img:
    image_path = file + i
    im = Image.open(image_path).convert('RGB')
    # Split into 3 channels
    r, g, b = im.split()
    im_r = np.array(r)
    diagram_h1_0 = CP.fit_transform(np.array(im_r)[None, :, :])
    y_betti_curves_h1_0 = BC.fit_transform(diagram_h1_0)
    data_RGB.append(np.reshape(y_betti_curves_h1_0, 100))

df0 = pd.DataFrame(data_RGB)
# df0["label"] = [1]*len(data)

df0.to_excel("B1_RC.xlsx")
df0