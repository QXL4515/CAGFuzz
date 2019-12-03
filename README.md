# CAGFuzz
CAGFuzz, a Coverage-guided Adversarial Generative Fuzzing testing approach for DL systems. The goal of the CAGFuzz is to maximize the neuron coverage and generate adversarial test examples as much as possible with minor perturbations for the target DNNs. It mainly consists of four folders：
* Coverage_Calculate
* CycleGAN
* model
* similarity


## Coverage_Calculate
This folder contains the code to calculate the neuron coverage. You can call the functions in the python file to run directly. An example is as follows:
```python
from Keras_coverage import NCoverage
from keras.models import load_model
model = load_model("./model/model_LeNet-1.h5")
coverage = NCoverage(model, 0.1)
img = Image.open('./datasets/cifar-10/coverage/img-0-frog.png')
img = np.array(img).astype('float32').reshape(-1, 32, 32, 3)
coverage.update_coverage(img)
covered, total, p = coverage.curr_neuron_cov()
print(covered, total, p)
```

## CycleGAN
Implementation of Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks.
Paper:https://arxiv.org/abs/1703.10593
#### Example
```
$ cd CycleGAN/
$ python CycleGAN_model.py
```
An example of the generated adversarial examples is as follows:

<img src="https://github.com/QXL4515/CAGFuzz/blob/master/picture/D1.jpg" width="290"/><img src="https://github.com/QXL4515/CAGFuzz/blob/master/picture/D2.jpg" width="290"/><img src="https://github.com/QXL4515/CAGFuzz/blob/master/picture/D3.jpg" width="290"/><img src="https://github.com/QXL4515/CAGFuzz/blob/master/picture/D4.jpg" width="290"/><img src="https://github.com/QXL4515/CAGFuzz/blob/master/picture/D5.jpg" width="290"/><img src="https://github.com/QXL4515/CAGFuzz/blob/master/picture/D6.jpg" width="290"/>


## model
This folder contains six neural networks for image recognition and a function for recording training loss, namely:
* LeNet-1
* LeNet-4
* LeNet-5
* VGG-16
* VGG-19
* ResNet-20
* LossHistory

If you want to train a LeNet-1 model of your own, please do as follows:
```
python LeNet-1.py
```
If you want to train a VGG-16 model of your own, please do as follows:
```
python VGG-16.py
```

## similarity
This folder contains two Python files, one is `vgg19_feature.py`, which is used to extract the depth features of pictures, the other is `utility.py`, which is used to compare the cosine similarity between the depth features of two pictures.

If you want to extract the depth features of an image, you can do this:
```python
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np
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
```
If you want to compare the cosine similarity between the depth features of two images, you can do this:
```python
from utility import get_cossimi
s1 = get_feature('1.png')
s2 = get_feature('0_1_000.png')
sim = get_cossimi(s1, s2)
print(sim)
```

## The overall process of realizing CAGFuzz

<img src="https://github.com/QXL4515/CAGFuzz/blob/master/picture/The overall process of realizing CAGFuzz.jpg" width="500"/>

The general process of CAG is shown in the figure above. The specific process can be as follows:
* First, we need to call `CycleGAN_,odel.py` in `CycleGAN` to train `AGE`.
* Then, the function of feature extraction is realized by `vgg19_feature.py` file in folder `similarity`.
* Finally, the implementation of neuron coverage needs file `Keras_coverage.py` under folder `Coverage_Calculate`.




