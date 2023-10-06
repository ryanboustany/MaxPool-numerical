import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import optuna
from optuna.trial import TrialState
import joblib

from models import *
from maxpool import *
from relu import *
from data_utils import *
from train import train, test

NTRIALS = 10
device = "cuda" if torch.cuda.is_available() else "cpu"

def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


def init(network, batch_norm, beta, alpha, precision):
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

    net = nn.DataParallel(net)

    return net

def objective(trial):
    global optimizer
    global dataset
    global epochs
    global beta
    global batch_norm
    global precision

    if dataset == "cifar10":
        trainloader, testloader = get_cifar10_loaders()
    elif dataset == "svhn":
        trainloader, testloader = get_svhn_loaders()
    elif dataset == "mnist":
        trainloader, testloader = get_mnist_loaders()

    net = init(network, batch_norm, beta, 0, precision)
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    if optimizer == "sgd":
        opt = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    elif optimizer == "adam":
        opt = optim.Adam(net.parameters(), lr=lr)

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train(net, opt, trainloader, precision)
        test_loss, test_acc = test(net, testloader, precision)
        trial.report(test_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return test_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lr search for Maxpool'()")
    parser.add_argument("--maxpool", type=float, default="0")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--network", type=str, default="vgg11")
    parser.add_argument("--batch_norm", type=boolean_string, default="False")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--trials", type=int, default=10, help="number of trials")
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--betas", nargs="+", type=float, default=[0, 1, 10, 100, 1000, 10000])
    args = vars(parser.parse_args())

    network = args["network"]
    optimizer = args["optimizer"]
    dataset = args["dataset"]
    epochs = args["epochs"]
    n_trials = args["trials"]
    batch_norm = args["batch_norm"]
    betas = args["betas"]
    precision = args["precision"]

    outdir = f"results/best_learning_rates/{dataset}/{network}/batch_norm_{batch_norm}/{optimizer}/bits_{precision}"
    os.makedirs(outdir, exist_ok=True)

    file_name = "best_lr.csv"
    print(f"OUTDIR: {outdir}")
    print(f"File: {file_name}")

    df = pd.DataFrame(columns=["maxpool", "lr", "accuracy"])

    for beta in betas:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        df = df.append({"maxpool": beta, "lr": trial.params["lr"], "accuracy": trial.value}, ignore_index=True)
        study_path = os.path.join(outdir, "study.pkl")
        joblib.dump(study, study_path)

    csv_path = os.path.join(outdir, file_name)
    df.to_csv(csv_path)
