from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np
import os
import scipy.io as sio
from utility import get_cossimi

def get_feature(img_dir):
    base_model = VGG19(weights='imagenet')
    model = Model(input=base_model.input, output=base_model.get_layer('fc2').output)
    img = image.load_img(img_dir, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    f = model.predict(x)
    print(f.shape)
    print(f)
    return f

s1 = get_feature('1.png')
s2 = get_feature('0_1_000.png')

sim = get_cossimi(s1, s2)
print(sim)


