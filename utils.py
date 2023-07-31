# coding: UTF-8
import collections
import os
import random

import torch
import numpy as np
import torchtext.vocab as Vocab
import torchtext
import time
from datetime import timedelta
import jieba
import pandas as pd
import torch.utils.data as Data
from PIL import Image
import torchvision.transforms as transforms

with open("Weibo_anxiety/data/cn_stopwords.txt", encoding='utf-8') as f:
    stopwords = f.read()
stopwords_list = stopwords.split('\n')
stopwords_list.append('焦虑症')
stopwords_list.append('\n')


# Setting random number seeds for easy reproduction
def set_seed():
    np.random.seed(27)
    torch.manual_seed(27)
    torch.cuda.manual_seed_all(27)
    torch.backends.cudnn.deterministic = True
    random.seed(27)
    os.environ["PYTHONHASHSEED"] = str(27)


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def cut_sentence(text):
    func = jieba.cut(text)
    return [word for word in func]


def cut_all_texts(content):
    cut_texts = []
    for text in content:
        cut = cut_sentence(text)
        cut_texts.append(cut)
    return cut_texts


def build_vocab(data, min_freq, stop=True):
    if stop:
        stopword_list = stopwords_list
    else:
        stopword_list = []

    content = data['content']
    cut_texts = cut_all_texts(content)
    avg_len = 0
    for i in cut_texts:
        avg_len += len(i)
    avg_len = avg_len // len(cut_texts)
    print(avg_len)
    counter = collections.Counter([tk for st in cut_texts for tk in st if tk not in stopword_list])
    v = torchtext.vocab.vocab(counter, min_freq=min_freq)
    return Vocab.Vocab(v), stopwords_list, avg_len


def preprocess_(data, img_path, input_size, vocab, pad_size=32, stop=True):
    # for nlp:
    # every text should be truncated or padded with 0 to avg_len
    def pad(x):
        return x[:pad_size] if len(x) > pad_size else x + [0] * (pad_size - len(x))

    cut_texts = cut_all_texts(data['content'])
    nlp_features = torch.tensor(
        [pad([vocab[word] for word in words if word not in stopwords_list]) for words in cut_texts])

    # for cv:
    # load image with id then transform to tensor with input_size
    trans = transforms.ToTensor()
    cv_features = []
    for index, row in data.iterrows():
        image_path = img_path + '/(' + str(row['id']) + ')/1.jpg'
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = image.resize((input_size, input_size))
        image = trans(image)
        image = torch.unsqueeze(image, 0)
        cv_features.append(image.numpy())
    cv_features = torch.tensor(np.concatenate(cv_features))

    # for behavior
    # choose hour, retweet, comment, subscribe,
    behavior_data = data.iloc[:, [2, 4, 5, 6]]
    behavior_features = torch.tensor(behavior_data.to_numpy())

    # for labels
    labels = torch.tensor([i for i in data['total']])

    # concatenate features [nlp, cv, behavior]
    features = torch.cat((nlp_features, cv_features.reshape(cv_features.shape[0], -1), behavior_features), 1)

    return features, labels


def build_datasets(config):
    print("build datasets...")
    data = pd.read_csv("Weibo_anxiety/data/final_data_11_13.csv")
    vocab, stopword_list, avg_len = build_vocab(data, min_freq=config.min_freq)

    config.pad_size = avg_len

    train_data = pd.read_excel(config.train_path)
    dev_data = pd.read_excel(config.dev_path)
    test_data = pd.read_excel(config.test_path)

    train_set = Data.TensorDataset(
        *preprocess_(train_data, config.image_path, config.input_size, vocab, config.pad_size))
    test_set = Data.TensorDataset(*preprocess_(test_data, config.image_path, config.input_size, vocab, config.pad_size))
    dev_set = Data.TensorDataset(*preprocess_(dev_data, config.image_path, config.input_size, vocab, config.pad_size))

    train_iter = Data.DataLoader(train_set, config.batch_size, shuffle=True, drop_last=True)
    test_iter = Data.DataLoader(test_set, config.batch_size, shuffle=True, drop_last=True)
    dev_iter = Data.DataLoader(dev_set, config.batch_size, shuffle=True, drop_last=True)

    return vocab, train_iter, dev_iter, test_iter
