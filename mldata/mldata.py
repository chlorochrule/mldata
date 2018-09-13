# -*- coding: utf-8 -*-

import numpy as np
from pathlib import Path
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data

DATA_DIR = '../database/'

def split_n_data(n_data, n_labels, same_size=True):
    n_class = len(n_labels)
    if same_size:
        n_sub_data = int(n_data / n_class)
        n_sub_datas = [n_sub_data] * n_class
        diff = n_data - sum(n_sub_datas)
        for i in np.arange(n_class)[np.argsort(n_labels)][:diff]:
            n_sub_datas[i] += 1
        return n_sub_datas
    else:
        ratio = n_data / np.sum(n_labels)
        n_sub_datas = [int(nd * ratio) for nd in n_labels]
        diff = n_data - sum(n_sub_datas)
        for i in np.arange(n_class)[np.argsort(n_labels)][:diff]:
            n_sub_datas[i] += 1
        return n_sub_datas


def select_data(data, label, n_data=None, i_dataset=0, same_size=True):
    if n_data is None:
        return shuffle(data, label)
    else:
        le = LabelEncoder()
        le.fit(label)
        label = le.transform(label)
        n_class = le.classes_.shape[0]
        data, label = shuffle(data, label)
        n_labels = np.bincount(label)
        n_sub_datas = split_n_data(n_data, n_labels, same_size=same_size)
        res_data = np.empty(shape=(0, data.shape[1]))
        res_label = np.empty(shape=(0,), dtype=np.int32)
        for i, n_sub_data in zip(range(n_class), n_sub_datas):
            sub_data = data[label==i]
            sub_data = sub_data[i_dataset*n_sub_data:(i_dataset+1)*n_sub_data]
            sub_label = np.full(shape=(sub_data.shape[0],), fill_value=i)
            res_data = np.concatenate((res_data, sub_data), axis=0)
            res_label = np.concatenate((res_label, sub_label), axis=0)
        return shuffle(res_data, res_label)


def load_datasets(datasets, is_tuning=False, tuning_rate=0.5):

    if datasets == 'mnist':
        mnist = fetch_mldata('MNIST original', data_home=DATA_DIR)
        data = mnist['data']
        label = mnist['target']

    elif datasets == 'eth80':
        data = np.empty(shape=(0, 28*28))
        label = np.empty(shape=(0,))
        objects = ['apple', 'car', 'cow', 'cup', 'dog', 'horse', 'pear', 'tomato']
        for i, obj in enumerate(objects):
            pathes = Path(DATA_DIR + 'eth80-cropped-perimg128') \
                .glob('{}*'.format(obj))
            for path in pathes:
                fpathes = path.glob('*.png')
                for fpath in fpathes:
                    img = Image.open(fpath).convert('L')
                    img_resized = img.resize((28, 28))
                    img_array = np.asarray(img_resized).reshape((1, -1))
                    data = np.append(data, img_array, axis=0)
                    label = np.append(label, i)

    elif datasets == 'coil20':
        data = np.empty(shape=(0, 28*28))
        label = np.empty(shape=(0,))
        for i in range(20):
            path = Path(DATA_DIR + 'coil-20-proc')
            fpathes = path.glob('obj{}__*.png'.format(i+1))
            cnt = 0
            for fpath in fpathes:
                cnt += 1
                img = Image.open(fpath).convert('L')
                img_resized = img.resize((28, 28))
                img_array = np.asarray(img_resized).reshape((1, -1))
                data = np.append(data, img_array, axis=0)
                label = np.append(label, i)

    elif datasets == 'fashion':
        fashion = input_data.read_data_sets(DATA_DIR + 'fashion-mnist',
                                            one_hot=False, validation_size=0)
        train_data = fashion.train.images  # Returns np.array
        train_labels = np.asarray(fashion.train.labels, dtype=np.int32)
        eval_data = fashion.test.images  # Returns np.array
        eval_labels = np.asarray(fashion.test.labels, dtype=np.int32)
        data = np.concatenate((train_data, eval_data))
        label = np.concatenate((train_labels, eval_labels))

    elif datasets == 'yale':
        data = np.empty(shape=(0, 320*243))
        label = np.empty(shape=(0,))
        for i in range(15):
            path = Path(DATA_DIR + 'yale-faces')
            fpathes = path.glob('subject{:02d}*.pgm'.format(i+1))
            cnt = 0
            for fpath in fpathes:
                cnt += 1
                img = Image.open(fpath).convert('L')
                img_array = np.asarray(img).reshape((1, -1))
                data = np.append(data, img_array, axis=0)
                label = np.append(label, i)

    else:
        raise Exception()

    data, label = shuffle(data, label)

    if is_tuning:
        thres = int(data.shape[0] * tuning_rate)
        return data[:thres, :], label[:thres], data[thres:, :], label[thres:]
    else:
        return data, label


def main():
    datasets = ['mnist', 'coil20', 'eth80', 'fashion', 'yale']
    for dataset in datasets:
        print(dataset)
        data, label = load_datasets(dataset)
        print(data.shape)
        print(label.shape)
        data, label = load_datasets(dataset)
        print(data.shape)
        print(label.shape)

if __name__ == '__main__':
    main()
