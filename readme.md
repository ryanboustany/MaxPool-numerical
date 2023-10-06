# Numerical reliability of nonsmooth autodiff : a MaxPool case study

This is an official implementation of the paper Numerical reliability of nonsmooth autodiff : a MaxPool case study. Please cite the paper and star this repo if you find this useful. Thanks!

## Dependencies

- [pytorch](https://pytorch.org) 
- torchvision
- optuna


## 📂 Repository Structure

### 🎈 Introduction experiment
- [Notebook](notebooks/first_xp.ipynb)

### 🎨 Figures

- **16-bit plot**: [Notebook](results/plot_16bits.ipynb)
- **32-bit plot**: [Notebook](results/plot_32bits.ipynb)
- **Mixed precision ImageNet**: [Notebook](results/plot_imagenet.ipynb)
- **Threshold analysis**: [Notebook](notebooks/threshold.ipynb)
- **Volume estimation**: [Notebook](notebooks/volume_estimation_by_architecture.ipynb)


### 📈 Section 4.2 Experiments
Execute the command below for experiments in Section 4.2:
```console
python train_with_best_lr.py --network [NETWORK] --dataset [DATASET] --batch_norm [BATCH_NORM] --epochs [EPOCHS]
``````
with ```[NETWORK]``` = mnist, vgg11 or resnet18 , ```[DATASET]``` = mnist, cifar10 or svhn and ```[BATCH_NORM]``` = True or False

Example: 
```console
python train_with_best_lr.py --network resnet18 --dataset cifar10 --batch_norm True --epochs 200 
```

### 📝 Additional Experiments
To run the additional experiments:
```console
python train_with_best_lr.py --network [NETWORK] --dataset[DATASET] --batch_norm [BATCH_NORM] --epochs 200
```

To run the imagenet experiments:
```console
python train_imagenet.py --dist-url 'tcp://127.0.0.1:9002' --dist-backend 'nccl' --maxpool [BETA] --multiprocessing-distributed --world-size 1 --rank 0 '{[IMAGENET_FOLDER_PATH]}'
```
