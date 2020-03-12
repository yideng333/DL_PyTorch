import torch
import time
import random
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report


def get_data_iter(train_data, train_label, batch_size):
    batch_len = len(train_data) // batch_size
    for i in range(batch_len):
        i = i * batch_size
        X = train_data[i: i + batch_size]
        Y = train_label[i: i + batch_size]
        yield X, Y


def padding_batch(batch_data, padding_value):
    batch_data.sort(key=lambda data: len(data), reverse=True)
    data_length = torch.LongTensor([len(data) for data in batch_data])
    # batch_data = [torch.LongTensor(i) for i in batch_data]
    train_data = torch.nn.utils.rnn.pad_sequence(batch_data, batch_first=True, padding_value=padding_value)
    return train_data, data_length


def shuffle_data(train_data, train_label):
    randnum = 666
    random.seed(randnum)
    random.shuffle(train_data)
    random.seed(randnum)
    random.shuffle(train_label)

    print(train_data, train_label)
    return train_data, train_label


def get_metric(preds, labels):
    precision = precision_score(labels, preds, average='macro')
    recall = recall_score(labels, preds, average='macro')
    f1 = f1_score(labels, preds, average='macro')
    report = classification_report(labels, preds)
    print(report)
    print('precision: {:.4f}  recall: {:.4f}  f1: {:.4f} \n'.format(precision, recall, f1))


def train_model(args, model, optimizer, train_data, train_label, test_data, test_label):
    for epoch in range(args.epochs):
        l_sum = 0.0
        start = time.time()
        train_pred = []
        train_target = []
        for sentence, targets in get_data_iter(train_data, train_label, args.batch_size):
            model.train()
            optimizer.zero_grad()
            sentence_in, sentence_len = padding_batch(sentence, 0)
            # print(sentence_in, sentence_len)
            tags, tags_len = padding_batch(targets, 0)
            # print(tags, tags_len)

            sentence_in, sentence_len = sentence_in.to(args.device), sentence_len.to(args.device)
            tags, tags_len = tags.to(args.device), tags_len.to(args.device)
            # Step 3. Run our forward pass.
            loss = model.neg_log_likelihood(sentence_in, tags, sentence_len)
            # print(loss)
            # print(model(sentence_in, sentence_len, concated=True))
            train_pred.extend(model(sentence_in, sentence_len, concated=True))
            # print(targets)
            for i in targets:
                train_target.extend(i.tolist())
            loss.backward()
            optimizer.step()
            l_sum += loss.item()
        avg_loss = l_sum/len(train_data)
        print('-' * 50)
        # print(train_pred)
        # print(train_target)
        print('Train Epoch [%d]: Average loss: %.6f, time: %.1f s' %(epoch, avg_loss, time.time() - start))
        get_metric(train_pred, train_target)
        evaluate(epoch, args, model, test_data, test_label)

    if args.save_model:
        torch.save(model.state_dict(), "./datasets/{}".format(args.save_model_name))


def evaluate(epoch, args, model, test_data, test_label):
    model.eval()
    start = time.time()
    with torch.no_grad():
        sentence_in, sentence_len = padding_batch(test_data, 0)
        tags, tags_len = padding_batch(test_label, 0)

        sentence_in, sentence_len = sentence_in.to(args.device), sentence_len.to(args.device)
        tags, tags_len = tags.to(args.device), tags_len.to(args.device)

        test_loss = model.neg_log_likelihood(sentence_in, tags, sentence_len)
        test_pred = model(sentence_in, sentence_len, concated=True)

        test_target = []
        for i in test_label:
            test_target.extend(i.tolist())

        avg_loss = test_loss / len(test_data)
        print('*' * 50)
        print('Test Epoch [%d]: Average loss: %.6f, time: %.1f s' % (epoch, avg_loss, time.time() - start))
        get_metric(test_pred, test_target)


def test_model(args, model, sentence):
    model.eval()
    sentence_in, sentence_len = padding_batch(sentence, 0)
    model.load_state_dict(torch.load("./datasets/{}".format(args.save_model_name), map_location=torch.device(args.device)))
    print('successfully load model {}'.format(args.save_model_name))
    with torch.no_grad():
        sentence_in, sentence_len = sentence_in.to(args.device), sentence_len.to(args.device)
        rlt = model(sentence_in, sentence_len, concated=False)
    return rlt
