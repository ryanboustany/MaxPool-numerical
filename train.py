import torch
import torch.nn as nn
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
scaler = torch.cuda.amp.GradScaler()

def train(net, optimizer, trainloader, precision):
    criterion = nn.CrossEntropyLoss()
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    with tqdm(trainloader, desc="batch_loop", leave=False) as pbar:
        for inputs, targets in pbar:
            optimizer.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)

            if precision == 16:
                inputs = inputs.half()
                assert inputs.dtype is torch.float16
            if precision == 64:
                inputs = inputs.double()

            if precision == 24:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix(train_loss=train_loss/len(trainloader), train_accuracy=100*correct/total)

    return train_loss/len(trainloader), correct/total


def test(net, testloader, precision):
    criterion = nn.CrossEntropyLoss()
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad(), tqdm(testloader, desc="batch_loop", leave=False) as pbar:
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            if precision == 16:
                inputs = inputs.half()
                assert inputs.dtype is torch.float16
            if precision == 64:
                inputs = inputs.double()

            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix(test_loss=test_loss/len(testloader), test_accuracy=100*correct/total)

    return test_loss/len(testloader), correct/total
