# DeepCore: A Comprehensive Library for Coreset Selection in Deep Learning [PDF](https://arxiv.org/pdf/2204.08499.pdf)


### Introduction
To advance the research of coreset selection in deep learning, we contribute a code library named **DeepCore**, an extensive and extendable code library, for coreset selection in deep learning, reproducing dozens of popular and advanced coreset selection methods and enabling a fair comparison of different methods in the same experimental settings. **DeepCore** is highly modular, allowing to add new architectures, datasets, methods and learning scenarios easily. It is built on PyTorch.   

### Coreset Methods
We list the methods in DeepCore according to the categories in our original paper, they are 1) geometry based methods Contextual Diversity (CD), Herding  and k-Center Greedy; 2) uncertainty scores; 3) error based methods Forgetting  and GraNd score ; 4) decision boundary based methods Cal  and DeepFool ; 5) gradient matching based methods Craig  and GradMatch ; 6) bilevel optimiza- tion methods Glister ; and 7) Submodularity based Methods (GC) and Facility Location (FL) functions. we also have Random selection as the baseline.

### Datasets
It contains a series of other popular computer vision datasets, namely MNIST, QMNIST, FashionMNIST, SVHN, CIFAR10, CIFAR100 and TinyImageNet and ImageNet.

### Models
They are two-layer fully connected MLP, LeNet , AlexNet, VGG, Inception-v3, ResNet, WideResNet and MobileNet-v3.

### Example
For Project A Task 2 we can run the following command to compute our model.
```sh
CUDA_VISIBLE_DEVICES=0 python -u main.py --fraction 1 --balance True --epochs 30 --dataset MNIST --data_path ~/datasets --num_exp 1 --workers 10 --optimizer SGD -se 10 --selection GraNd --model InceptionV3 --lr 0.1 -sp ./result --batch 128
```


### References

1. Agarwal, S., Arora, H., Anand, S., Arora, C.: Contextual diversity for active learning. In: ECCV. pp. 137–153. Springer (2020)
2. Coleman, C., Yeh, C., Mussmann, S., Mirzasoleiman, B., Bailis, P., Liang, P., Leskovec, J., Zaharia, M.: Selection via proxy: Efficient data selection for deep learning. In: ICLR (2019)
3. Ducoffe, M., Precioso, F.: Adversarial active learning for deep networks: a margin based approach. arXiv preprint arXiv:1802.09841 (2018)
4. Iyer, R., Khargoankar, N., Bilmes, J., Asanani, H.: Submodular combinatorial information measures with applications in machine learning. In: Algorithmic Learning Theory. pp. 722–754. PMLR (2021)
5. Killamsetty, K., Durga, S., Ramakrishnan, G., De, A., Iyer, R.: Grad-match: Gradient matching based data subset selection for efficient deep model training. In: ICML. pp. 5464–5474 (2021)
6. Killamsetty, K., Sivasubramanian, D., Ramakrishnan, G., Iyer, R.: Glister: Generalization based data subset selection for efficient and robust learning. In: Proceedings of the AAAI Conference on Artificial Intelligence (2021)
7. Margatina, K., Vernikos, G., Barrault, L., Aletras, N.: Active learning by acquiring contrastive examples. arXiv preprint arXiv:2109.03764 (2021)
8. Mirzasoleiman, B., Bilmes, J., Leskovec, J.: Coresets for data-efficient training of machine learning models. In: ICML. PMLR (2020)
9. Paul, M., Ganguli, S., Dziugaite, G.K.: Deep learning on a data diet: Finding important examples early in training. arXiv preprint arXiv:2107.07075 (2021)
10. Sener, O., Savarese, S.: Active learning for convolutional neural networks: A coreset approach. In: ICLR (2018)
11. Toneva, M., Sordoni, A., des Combes, R.T., Trischler, A., Bengio, Y., Gordon, G.J.: An empirical study of example forgetting during deep neural network learning. In: ICLR (2018)
12. Welling, M.: Herding dynamical weights to learn. In: Proceedings of the 26th Annual International Conference on Machine Learning. pp. 1121–1128 (2009)


