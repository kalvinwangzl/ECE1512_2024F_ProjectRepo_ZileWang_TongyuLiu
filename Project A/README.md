# Project A: Dataset Distillation
The project consists of two major parts: Task 1 and Task 2. 

Note that some of the models and data need to be trained or downloaded before training certain models. Our code support all the training and downloading needed in the Tasks except for a provided mhist_dataset folder.

Also note that although some of our codes come from open source code online, we've made essential editation to them to apply on our tasks, which means the original codes probably won't work well if directed used.

## Task 1
In Task 1 we need to use the code in [naive](./naive) and [DataDAM](./DataDAM)

Code in [naive](./naive) are for training the models on original datasets by running [train_MNIST_or_CIFAR.py](./naive/train_MNIST_or_CIFAR.py) and [train_MHIST.py](./naive/train_MHIST.py)

Code in [DataDAM](./DataDAM) are for training the models using DataDAM methods that generates the distillated images, which can be achieved by running [main_DataDAM.py](./DataDAM/main_DataDAM.py)

## Task 2
In Task 2 we need to use the code in [DeepCore](./DeepCore), [PAD](./PAD) and [EDC](./EDC)

### PAD
To utilize PAD models, we need to first train a Deepcore model that will be used in PAD training. Instructions for training DeepCore and PAD are included in the README.md in each folder.

### EDC
Further instructions on how to train and evaluate EDC model are contained in the README.md in the folder.

# Reference
[DataDAM](https://github.com/DataDistillation/DataDAM), [DeepCore](https://github.com/patrickzh/deepcore), [Prioritize Alignment in Dataset Distillation](https://github.com/NUS-HPC-AI-Lab/PAD), [Efficient-Dataset-Condensation
](https://github.com/snu-mllab/Efficient-Dataset-Condensation)
