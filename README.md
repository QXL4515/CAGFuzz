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
![](https://github.com/QXL4515/CAGFuzz/tree/master/picture/D1.jpg)
![](https://github.com/QXL4515/CAGFuzz/tree/master/picture/D2.jpg)
![](https://github.com/QXL4515/CAGFuzz/tree/master/picture/D3.jpg)
![](https://github.com/QXL4515/CAGFuzz/tree/master/picture/D4.jpg)
![](https://github.com/QXL4515/CAGFuzz/tree/master/picture/D5.jpg)
![](https://github.com/QXL4515/CAGFuzz/tree/master/picture/D6.jpg)
