# CAGFuzz
CAGFuzz, a Coverage-guided Adversarial Generative Fuzzing testing approach for DL systems. The goal of the CAGFuzz is to maximize the neuron coverage and generate adversarial test examples as much as possible with minor perturbations for the target DNNs. It mainly consists of four folders：
* Coverage_Calculate
* CycleGAN
* model
* similarity
## Coverage_Calculate
This folder contains the code to calculate the neuron coverage. You can call the functions in the python file to run directly. An example is as follows:
'<from Keras_coverage import NCoverage>'
'<from keras.models import load_model>'
