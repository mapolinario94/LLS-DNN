# LLS: Local Learning Rule for Deep Neural Networks Inspired by Neural Activity Synchronization

This source code implements LLS, a local learning rule for training deep neural networks (DNNs), and replicates the experimental results obtained on the MNIST, FashionMNIST, CIFAR10, CIFAR100, TinyIMAGENET, IMAGENETTE, and Visual Wake Words (VWW) datasets. Its primary purpose is to aid in understanding the methodology and reproduce essential results. 
## How to Use

1. Install the required dependencies listed in `requirements.txt`. 
2. Use the following command to run an experiment:

    ```shell
    python main.py --param-name param_value
    ```

    A description of each parameter is provided in `main.py`.

## Reproducing Results in the Paper

To ensure reproducibility, we have provided a bash script (`./script.sh`) with all the commands used to obtain the results reported in Tables 1, 2, and 3 of the paper.

## Acknowledgments

We have utilized several publicly available repositories for specific functionalities in our work:

- For Schedule-Free SGD and AdamW optimizers, we used the code implementation from: [schedule_free](https://github.com/facebookresearch/schedule_free).
- For DFA experiments, we used the code implementation from: [DirectRandomTargetProjection](https://github.com/ChFrenkel/DirectRandomTargetProjection).
- For downloading the VWW dataset, we used the repository: [visualwakewords](https://github.com/Mxbonn/visualwakewords).
- For downloading the IMAGENETTE dataset, we used the repository: [imagenette](https://github.com/fastai/imagenette).
