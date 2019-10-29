import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

train_images_dir = "/home/rex/datasets/severstal-steel-defect-detection/"


def prepare_data(num_splits=1):
    data = pd.read_csv(train_images_dir + "/train.csv", header=0)
    data['image'] = data['ImageId_ClassId'].apply(
        lambda x: train_images_dir + "train_images/" + str(x.split("_")[0]))
    data['class_id'] = data['ImageId_ClassId'].apply(lambda x: x.split("_")[1])
    data2 = data.pivot(index='image',
                       columns='class_id',
                       values='EncodedPixels').reset_index()
    data2 = data2.fillna('')
    data2['label_cnt'] = data2.apply(lambda row: int(len(row['1']) > 0) + int(
        len(row['2']) > 0) + int(len(row['3']) > 0) + int(len(row['4']) > 0),
                                     axis=1)
    data2['labels_present'] = data2.apply(lambda row: calculate_labels(row),
                                          axis=1)
    data2 = data2[data2['label_cnt'] > 0]

    data2['labels_present'] = data2.apply(lambda row: calculate_labels(row),
                                          axis=1)
    x1 = data2[data2['labels_present'] == (2, 4)].index.values
    x2 = data2[data2['labels_present'] == (1, 2, 3)].index.values
    _labels_remove = np.concatenate([x1, x2], axis=0)
    data2 = data2.drop(_labels_remove)

    sss = StratifiedShuffleSplit(n_splits=num_splits, test_size=0.15)
    _splits = sss.split(data2, data2['labels_present'])

    _split_data = []
    for train_idx, val_idx in _splits:
        train_data_final = data2.iloc[train_idx]
        val_data_final = data2.iloc[val_idx]

        train_weights, val_weights = calculate_sample_weights(
            train_data_final), calculate_sample_weights(val_data_final)
        _split_data.append(
            [train_data_final, val_data_final, train_weights, val_weights])
    return _split_data


def rle2mask(rle_string):
    """
    returns the mask for one rle string
    :param rle_string:
    :return:
    """
    mask_shape = (256, 1600)
    mask = np.zeros((mask_shape[0] * mask_shape[1]))

    rle_string = str(rle_string.numpy().decode('utf8'))

    if len(rle_string) <= 0:
        return mask.reshape(mask_shape, order='F')
    else:
        start_pos = [int(_) for _ in rle_string.split(" ")[::2]]
        enc_length = [int(_) for _ in rle_string.split(" ")[1::2]]

        for idx, _pos in enumerate(start_pos):
            mask[_pos - 1:_pos - 1 + enc_length[idx]] = 1.0
        mask = mask.reshape(mask_shape, order='F')
        assert len(np.argwhere(mask > 0)) > 0
        return mask


def read_dataset(img, _masks):
    c1, c2, c3, c4 = _masks
    c1_mask = np.expand_dims(rle2mask(c1), axis=-1)
    c2_mask = np.expand_dims(rle2mask(c2), axis=-1)
    c3_mask = np.expand_dims(rle2mask(c3), axis=-1)
    c4_mask = np.expand_dims(rle2mask(c4), axis=-1)

    c0_mask = np.zeros_like(c1_mask)

    final_mask = np.concatenate([c0_mask, c1_mask, c2_mask, c3_mask, c4_mask],
                                axis=-1)

    max_labels = np.argmax(final_mask, axis=-1)
    tmp_mask = np.zeros_like(final_mask)
    tmp_mask[np.arange(final_mask.shape[0])[:, None],
             np.arange(final_mask.shape[1]), max_labels] = 1.0

    assert tmp_mask.shape == final_mask.shape
    return img, tmp_mask


def read_img(img_path, label):
    return tf.io.decode_jpeg(tf.io.read_file(img_path)), tf.convert_to_tensor(
        tf.cast(label, dtype=tf.float32), dtype=tf.float32)


def read_img_resize(img_path, label):
    return tf.image.resize(tf.io.decode_jpeg(tf.io.read_file(img_path)),
                           (224, 224)), tf.convert_to_tensor(tf.cast(
                               label, dtype=tf.float32),
                                                             dtype=tf.float32)


def calculate_sample_weights(df):
    cnt1_ = np.array(
        list(
            map(
                lambda x: 0 if x == '' else np.sum(
                    np.array(x.split(' ')[1::2]).astype(int)),
                df['1'].values))).sum()
    cnt2_ = np.array(
        list(
            map(
                lambda x: 0 if x == '' else np.sum(
                    np.array(x.split(' ')[1::2]).astype(int)),
                df['2'].values))).sum()
    cnt3_ = np.array(
        list(
            map(
                lambda x: 0 if x == '' else np.sum(
                    np.array(x.split(' ')[1::2]).astype(int)),
                df['3'].values))).sum()
    cnt4_ = np.array(
        list(
            map(
                lambda x: 0 if x == '' else np.sum(
                    np.array(x.split(' ')[1::2]).astype(int)),
                df['4'].values))).sum()
    x1 = np.array([cnt1_, cnt2_, cnt3_, cnt4_])

    total_mask_pixels = np.floor(df.shape[0] / 4) * 256 * 1600
    cnt_0 = total_mask_pixels - np.sum(x1)  # cnt0 pixels

    x2 = np.concatenate([[cnt_0], x1], axis=0)
    x3 = x2.sum() / x2
    x4 = x3 / x3.max()
    return list(x4)


def calculate_labels(row):
    _labels = []
    if len(row['1']) > 0:
        _labels.append(1)
    if len(row['2']) > 0:
        _labels.append(2)
    if len(row['3']) > 0:
        _labels.append(3)
    if len(row['4']) > 0:
        _labels.append(4)
    return tuple(_labels)


def tile_dataset(x, y):
    x_crop_boxes = []
    y_crop_boxes = []

    for _start_width in range(0, 1600, 200):
        x_crop_boxes.append(
            tf.image.resize(
                tf.image.crop_to_bounding_box(x, 0, _start_width, 128, 200),
                (224, 224)))
        y_crop_boxes.append(
            tf.image.resize(
                tf.image.crop_to_bounding_box(y, 0, _start_width, 128, 200),
                (224, 224)))

    return tf.data.Dataset.from_tensor_slices((x_crop_boxes, y_crop_boxes))


def prepare_data_classifier():
    data = pd.read_csv(train_images_dir + "/train.csv", header=0)
    data['image'] = data['ImageId_ClassId'].apply(
        lambda x: train_images_dir + "train_images/" + str(x.split("_")[0]))
    data['class_id'] = data['ImageId_ClassId'].apply(lambda x: x.split("_")[1])
    data2 = data.pivot(index='image',
                       columns='class_id',
                       values='EncodedPixels').reset_index()
    data2 = data2.fillna('')
    data2['label_cnt'] = data2.apply(lambda row: int(len(row['1']) > 0) + int(
        len(row['2']) > 0) + int(len(row['3']) > 0) + int(len(row['4']) > 0),
                                     axis=1)
    data2['labels_present'] = data2.apply(lambda row: calculate_labels(row),
                                          axis=1)

    data2['labels_present2'] = data2['labels_present'].apply(
        lambda x: 0 if len(x) == 0 else 1)
    # data2 = data2[data2['label_cnt'] == 0]

    # These combinations occur rarely, so remove
    x1 = data2[data2['labels_present'] == (2, 4)].index.values
    x2 = data2[data2['labels_present'] == (1, 2, 3)].index.values
    _labels_remove = np.concatenate([x1, x2], axis=0)
    data2 = data2.drop(_labels_remove)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15)
    _splits = sss.split(data2, data2['labels_present2'])

    for train_idx, val_idx in _splits:
        train_data_final = data2.iloc[train_idx]
        val_data_final = data2.iloc[val_idx]

        train_weights, val_weights = calculate_sample_weights(
            train_data_final), calculate_sample_weights(val_data_final)
    return train_data_final, val_data_final, train_weights, val_weights
