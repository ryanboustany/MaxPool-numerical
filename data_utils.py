import torchvision
import torchvision.transforms as transforms
import torch
import os 
import sys

dir_path = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(dir_path, 'data')

def get_cifar10_train_loader():
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    return torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=8)

def get_cifar10_test_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    return torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

def get_cifar10_loaders():
    trainloader = get_cifar10_train_loader()
    testloader = get_cifar10_test_loader()
    return trainloader, testloader

def get_cifar100_train_loader():
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=False, transform=transform)
    return torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=8)

def get_cifar100_test_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=False, transform=transform)
    return torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

def get_cifar100_loaders():
    trainloader = get_cifar100_train_loader()
    testloader = get_cifar100_test_loader()
    return trainloader, testloader

def get_svhn_loaders():
    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    trainset = torchvision.datasets.SVHN(root=data_dir, split='extra', download=False, transform=transform)
    testset = torchvision.datasets.SVHN(root=data_dir, split='test', download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=16)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=16)
    return trainloader, testloader

def get_mnist_loaders(**kwargs):
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        ])

    trainset = torchvision.datasets.MNIST(root=data_dir, train=True, download=False, transform=transform)
    testset = torchvision.datasets.MNIST(root=data_dir, train=False, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
    return trainloader, testloader
            

def get_fashion_mnist_loaders(**kwargs):
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        ])

    trainset = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=False, transform=transform)
    testset = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
    return trainloader, testloader



def get_kmnist_loaders(**kwargs):
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        ])

    trainset = torchvision.datasets.KMNIST(root=data_dir, train=True, download=False, transform=transform)
    testset = torchvision.datasets.KMNIST(root=data_dir, train=False, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
    return trainloader, testloader