import torch
import time


# 训练函数
def train(args, model, loss, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    train_loss = 0
    start = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(args.device), target.to(args.device)
        optimizer.zero_grad()
        output = model(data)
        l = loss(output, target)
        l.backward()
        optimizer.step()
        train_loss += l.item() # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), l.item()))
    train_loss /= len(train_loader.dataset)
    print('Train Epoch: {} Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Time: {:.1f} sec\n'.format(
        epoch, train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset), time.time() - start))


# 测试函数
def test(args, model, loss, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)
            test_loss += loss(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test Epoch: {} Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        epoch, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

