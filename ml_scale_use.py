"""
    加载模型 并使用模型
"""
import numpy as np
from PIL import ImageEnhance, Image
import matplotlib.pyplot as mp
import pickle
import sklearn

from lib.share import shareInfo

x_scale = 1 #pix to cm
y_scale = 1

def enhance(image):
    # 亮度增强
    enh_bri = ImageEnhance.Brightness(image)
    brightness = 0.3
    image_brightened = enh_bri.enhance(brightness)
    # image_brightened.show()

    # 色度增强
    enh_col = ImageEnhance.Color(image_brightened)
    color = 2
    image_colored = enh_col.enhance(color)
    # image_colored.show()

    # 对比度增强
    enh_con = ImageEnhance.Contrast(image_colored)
    contrast = 5
    image_contrasted = enh_con.enhance(contrast)
    # image_contrasted.show()

    # 锐度增强
    enh_sha = ImageEnhance.Sharpness(image_contrasted)
    sharpness = 3.0
    image_sharped = enh_sha.enhance(sharpness)
    # image_sharped.show()

    return image_sharped
def pix_scale(x,y,size):
    global x_scale,y_scale,x_prd_y,y_prd_y
    # 加载模型 使用模型
    x = np.array(x).reshape(1, -1)
    y = np.array(y).reshape(1, -1)

    with open('./linear2_x.pkl', 'rb') as f:
        model = pickle.load(f)
        x_prd_y= model.predict(x)
        #x_scale = size / x_prd_y
        #shareInfo.x_scale = x_scale
    with open('./linear2_y.pkl', 'rb') as f:
        model = pickle.load(f)
        y_prd_y = model.predict(y)
        #y_scale = size / y_prd_y
        #shareInfo.y_scale = y_scale
        #print('ml_scale_use result:','x_scale=', x_scale, 'y_scale=', y_scale)


