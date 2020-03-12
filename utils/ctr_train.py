import torch
import time
import torch.nn.functional as F


# 训练函数
def train_model(args, model, loss, train_loader, test_loader, optimizer):
    model.train()
    train_result = []
    valid_result = []
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        train_loss = 0.0
        y_pred = []
        y = []
        epoch_begin_time = time.time()
        batch_begin_time = time.time()

        for batch_idx, (target, X_i, X_v) in enumerate(train_loader):
        # for batch_idx, (target, X_i_num, X_v_num, X_i_cat, X_v_cat) in enumerate(train_loader):
            y.extend(target.data.numpy())
            target, X_i, X_v = target.to(args.device), X_i.to(args.device), X_v.to(args.device)
            # target, X_i_num, X_v_num, X_i_cat, X_v_cat = target.to(args.device), X_i_num.to(args.device), \
            #                                              X_v_num.to(args.device), X_i_cat.to(args.device), \
            #                                              X_v_cat.to(args.device)

            optimizer.zero_grad()
            outputs = model(X_i, X_v)
            # outputs = model(X_i_num, X_v_num, X_i_cat, X_v_cat)
            pred = F.sigmoid(outputs).cpu()
            y_pred.extend(pred.data.numpy())
            l = loss(outputs, target)
            l.backward()
            optimizer.step()

            train_loss += l.item()
            total_loss += l.item()
            if batch_idx > 0 and batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t loss: {:.6f} time: {:.2f} s'.format
                      (epoch, batch_idx * len(target), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                       train_loss, time.time() - batch_begin_time))
                train_loss = 0.0
                batch_begin_time = time.time()

        train_metric = args.eval_metric(y, y_pred)
        train_result.append(train_metric)
        total_loss /= len(train_loader.dataset)
        print('*' * 50)
        print('Train Epoch [%d]: Average loss: %.6f, metric: %.6f time: %.1f s' %
              (epoch, total_loss, train_metric, time.time() - epoch_begin_time))

        valid_loss, valid_metric = eval_valid_set(args, model, loss, test_loader)
        valid_result.append(valid_metric)
        print('Valid Epoch [{}]: Average loss: {:.4f}, metric: {}'.format(epoch, valid_loss, valid_metric))
        print('*' * 50)

        if args.save_model:
            torch.save(model.state_dict(), "model_epoch_{}.pt".format(epoch))

        # if is_valid and ealry_stopping and self.training_termination(valid_result):
        #     print("early stop at [%d] epoch!" % (epoch + 1))
        #     break


def eval_valid_set(args, model, loss, test_loader):
    model.eval()
    valid_loss = 0.0
    y_pred = []
    y = []

    with torch.no_grad():
        for batch_idx, (target, X_i, X_v) in enumerate(test_loader):
        # for batch_idx, (target, X_i_num, X_v_num, X_i_cat, X_v_cat) in enumerate(test_loader):
            y.extend(target.data.numpy())
            target, X_i, X_v = target.to(args.device), X_i.to(args.device), X_v.to(args.device)
            # target, X_i_num, X_v_num, X_i_cat, X_v_cat = target.to(args.device), X_i_num.to(args.device), \
            #                                              X_v_num.to(args.device), X_i_cat.to(args.device), \
            #                                              X_v_cat.to(args.device)
            outputs = model(X_i, X_v)
            # outputs = model(X_i_num, X_v_num, X_i_cat, X_v_cat)
            pred = F.sigmoid(outputs).cpu()
            y_pred.extend(pred.data.numpy())
            l = loss(outputs, target)
            valid_loss += l.item()

    valid_metric = args.eval_metric(y, y_pred)
    valid_loss /= len(test_loader.dataset)
    return valid_loss, valid_metric
