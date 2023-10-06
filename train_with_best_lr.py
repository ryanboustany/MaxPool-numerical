import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import csv

from models import *
from relu import *
from maxpool import *
from data_utils import *
from train import *
import torch.backends.cudnn as cudnn


def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"

def init(network, optimizer, batch_norm, beta, alpha, precision, device, lr, n_epochs):
    if network == "vgg11":
        net = VGG("VGG11", batch_norm=batch_norm, maxpool_fn=lambda: MaxPool2DBeta(beta), relu_fn=lambda: ReLUAlpha(alpha))
    elif network == "vgg19":
        net = VGG("VGG19", batch_norm=batch_norm, maxpool_fn=lambda: MaxPool2DBeta(beta), relu_fn=lambda: ReLUAlpha(alpha))
    elif network == "resnet18":
        if batch_norm:
            net = resnet18(norm_layer=nn.BatchNorm2d, maxpool_fn=lambda: MaxPool2DBeta(beta), relu_fn=lambda: ReLUAlpha(alpha))
        else:
            net = resnet18(norm_layer=nn.Identity, maxpool_fn=lambda: MaxPool2DBeta(beta), relu_fn=lambda: ReLUAlpha(alpha))
    elif network == "resnet50":
        if batch_norm:
            net = resnet50(norm_layer=nn.BatchNorm2d, maxpool_fn=lambda: MaxPool2DBeta(beta), relu_fn=lambda: ReLUAlpha(alpha))
        else:
            net = resnet50(norm_layer=nn.Identity, maxpool_fn=lambda: MaxPool2DBeta(beta), relu_fn=lambda: ReLUAlpha(alpha))
    elif network == "densenet121":
        if batch_norm:
            net = densenet121(norm_layer=nn.BatchNorm2d, maxpool_fn=lambda: MaxPool2DBeta(beta), relu_fn=lambda: ReLUAlpha(alpha))
        else:
            net = densenet121(norm_layer=nn.Identity, maxpool_fn=lambda: MaxPool2DBeta(beta), relu_fn=lambda: ReLUAlpha(alpha))
    elif network == "lenet":
        net = LeNet5(maxpool_fn=lambda: MaxPool2DBeta(beta), relu_fn=lambda: ReLUAlpha(alpha), batch_norm=batch_norm)
    else:
        raise ValueError("Invalid network name.")

    net = net.to(device)

    if precision == 16:
        net = net.half()
    elif precision == 64:
        net = net.double()

    if optimizer == "sgd":
        opt = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    elif optimizer == "adam":
        opt = optim.Adam(net.parameters(), lr=lr)
        
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    return net, opt, scheduler

def main(args):
    network = args["network"]
    optimizer = args["optimizer"]
    dataset = args["dataset"]
    epochs = args["epochs"]
    batch_norm = args["batch_norm"]
    nb_experiment = args["nb_experiment"]
    precision = args["precision"]
    n_epochs = args['epochs']

    outdir = f"results/train/{dataset}/{network}/batch_norm_{batch_norm}/{optimizer}/bits_{precision}"
    os.makedirs(outdir, exist_ok=True)

    filename = os.path.join(outdir, "results.csv")

    fieldnames = ["run_id", "epoch", "train_loss", "train_accuracy", "test_loss", "test_accuracy", "maxpool", "batch_norm"]

    with open(filename, mode="w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

    print(f"File: {filename}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if dataset == "cifar10":
        trainloader, testloader = get_cifar10_loaders()
    elif dataset == "svhn":
        trainloader, testloader = get_svhn_loaders()
    elif dataset == "mnist":
        trainloader, testloader = get_mnist_loaders()

    best_lrs = pd.read_csv(os.path.join("results/best_learning_rates", dataset, network, f"batch_norm_{batch_norm}", optimizer, f"bits_{precision}", "best_lr.csv"))
    maxpool_values = best_lrs.maxpool.values
    learning_rates = []
    for i in range(len(maxpool_values)):
        learning_rates.append(best_lrs.lr.values[0])

    for maxpool_value, lr in tqdm(zip(maxpool_values, learning_rates), desc="maxpool_value", leave=False):
        for k in tqdm(range(nb_experiment), desc="run_loop", leave=False):
            net, opt, scheduler = init(network, optimizer, batch_norm, maxpool_value, 0, precision, device, lr, n_epochs)
            for epoch in tqdm(range(epochs), desc="epoch_loop", leave=False):
                train_loss, train_acc = train(net, opt, trainloader, precision)
                test_loss, test_acc = test(net, testloader, precision)
                with open(filename, mode="a", newline="") as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    writer.writerow({"run_id": k, "epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc, "test_loss": test_loss, "test_accuracy": test_acc, "maxpool": maxpool_value, "batch_norm": batch_norm})
                scheduler.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch maxpool experiment impact of different values for MaxPool'() with best lr")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--nb_experiment", type=int, default=10, help="number of independent runs for each configuration")
    parser.add_argument("--batch_norm", type=boolean_string, default="False")
    parser.add_argument("--network", type=str, default="vgg11")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--precision", type=int, default=32)

    args = vars(parser.parse_args())
    main(args)