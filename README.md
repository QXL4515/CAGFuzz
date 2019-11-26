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
