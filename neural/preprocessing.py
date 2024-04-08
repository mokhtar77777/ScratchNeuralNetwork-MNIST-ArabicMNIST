import numpy as np


def to_one_hot(labels):
    labels_arr = np.array(labels)

    one_hot_row_num = labels_arr.shape[0]
    one_hot_col_num = np.max(labels_arr) + 1
    one_hot_arr = np.zeros(shape=(one_hot_row_num, int(one_hot_col_num)))

    one_hot_arr[np.arange(one_hot_row_num), labels_arr.astype(np.int_)] = 1

    return one_hot_arr


def shuffle_dataset(data, labels):
    data = np.array(data)
    labels = np.array(labels)

    m = data.shape[0]

    random_idx = np.random.permutation(m)

    new_data = data[random_idx]
    new_labels = labels[random_idx]

    return new_data, new_labels


def mini_batch_generator(data, labels, batch_size, shuffle=True):
    data = np.array(data)
    labels = np.array(labels)

    m = data.shape[0]

    if shuffle:
        data, labels = shuffle_dataset(data, labels)

    batches_num = m // batch_size

    for b in range(batches_num):
        start = b * batch_size
        end = (b+1) * batch_size
        mini_batch_data = data[start:end]
        mini_batch_lbl = labels[start:end]

        yield mini_batch_data, mini_batch_lbl

    start = batch_size * batches_num
    if start != m:
        final_data = data[start:]
        final_lbl = labels[start:]

        yield final_data, final_lbl


def data_generator(data, batch_size):
    data = np.array(data)

    m = data.shape[0]
    batches_num = m // batch_size

    for b in range(batches_num):
        start = b * batch_size
        end = (b+1) * batch_size
        mini_batch_data = data[start:end]

        yield mini_batch_data

    start = batch_size * batches_num
    if start != m:
        final_data = data[start:]

        yield final_data
