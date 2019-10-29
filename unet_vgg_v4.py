import tensorflow as tf
from tensorflow.keras import backend as K
import pandas as pd
import numpy as np
import os
from tensorflow.keras.applications.vgg16 import VGG16
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, BatchNormalization, ReLU, Input, ZeroPadding2D, Dropout, UpSampling2D, Concatenate, Cropping2D
from loss import *
from metric import *

## Adds dropout to conv layers + exp17
## patience changed from 10 to 5

exp = "exp23_vgg2/"
model_dir = "/home/rex/workspace/kaggle-severstal-defects/checkpoints/" + exp
train_images_dir = "/home/rex/datasets/severstal-steel-defect-detection/"

log_dir = os.path.join(model_dir, 'trianing.csv')
if not os.path.exists(model_dir):
    os.mkdir(model_dir)


def conv_block(input, filters, is_final):
    if not is_final:
        x = MaxPool2D(1, (2, 2))(input)
    else:
        x = input

    x = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    # if filters <= 256:
    #     x = Dropout(0.2)(x)
    x = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    # if filters >= 256:
    #     x = Dropout(0.2)(x)
    tf.print("End of each down-conv: ", x.shape)
    return x


def upconv_block(input, conv_op, filters):
    x1 = Conv2DTranspose(int(filters / 2), (2, 2), strides=(2, 2))(input)
    x = Concatenate()([x1, conv_op])
    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(0.2)(x)
    # if filters <= 256:
    #     x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    # if filters <= 256:
    #     x = Dropout(0.2)(x)
    return x


def unet():
    input = Input(shape=(224, 224, 3))
    vgg = VGG16(weights='imagenet', include_top=False)  #
    for layer in vgg.layers:
        layer.trainable = False

    vgg_model = tf.keras.Model(vgg.input, [
        vgg.get_layer('block5_pool').output,
        vgg.get_layer('block5_conv3').output,
        vgg.get_layer('block4_conv3').output,
        vgg.get_layer('block3_conv3').output,
        vgg.get_layer('block2_conv2').output,
        vgg.get_layer('block1_conv2').output
    ])

    b5_pool, b5_c3, b4_c3, b3_c3, b2_c2, b1_c3 = vgg_model(input)
    last_conv = Conv2D(1024, (3, 3), activation='relu',
                       padding='same')(b5_pool)

    print("las_conv shape: ", last_conv.shape)

    upconv_block1 = upconv_block(
        last_conv, b5_c3,
        512)  ## b5_pool= (7, 7, 512), last_conv = (7, 7, 1024) after

    upconv_block2 = upconv_block(upconv_block1, b4_c3, 512)  ## 28 after
    upconv_block3 = upconv_block(upconv_block2, b3_c3, 256)  ## 56
    upconv_block4 = upconv_block(upconv_block3, b2_c2, 128)  ## 112
    upconv_block5 = upconv_block(upconv_block4, b1_c3, 64)  ## 224

    # upconv_block4 = upconv_block(upconv_block4, down_conv.get_layer('block1_conv2').output, ((1, 1), (1, 1)),
    #                              64)  ## 224
    # fop1 = UpSampling2D()(upconv_block5)
    final_output = Conv2D(5, (1, 1), activation="softmax")(upconv_block5)
    print("final output shape: ", final_output.shape)
    model = tf.keras.Model(input, final_output)
    # model output is (N, 128 * 800, 5) ## (N, 128 , 800, 5)
    return model


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


def prepare_data():
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

    #train_data, val_data = train_test_split(data2, test_size=0.20, random_state=42)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15)
    _splits = sss.split(data2, data2['labels_present'])

    for train_idx, val_idx in _splits:
        train_data_final = data2.iloc[train_idx]
        val_data_final = data2.iloc[val_idx]

    train_weights, val_weights = calculate_sample_weights(
        train_data_final), calculate_sample_weights(val_data_final)
    return train_data_final, val_data_final, train_weights, val_weights


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
        label, dtype=tf.float32)


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


def get_dataset(data):
    ds = tf.data.Dataset.from_tensor_slices(
        (data['image'].values, (data['1'].values, data['2'].values,
                                data['3'].values, data['4'].values)))
    ds = ds.map(lambda x, y: tf.py_function(read_dataset, [x, y],
                                            [tf.string, tf.float32]))
    ds = ds.map(lambda x, y: read_img(x, y))
    ds = ds.flat_map(lambda x, y: tile_dataset(x, y))
    ds = ds.shuffle(buffer_size=100, seed=40)
    # ds = ds.map(lambda x, y: _hot_encode(x, y))
    ds = ds.batch(6)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds


def tmp_dice(y_true, y_pred):
    smooth = 1
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return 1 - ((2. * intersection + smooth) /
                (K.sum(y_true) + K.sum(y_pred) + smooth))


def scheduler(epoch):
    if epoch < 6:
        return 0.0001
    else:
        return 0.0001 * tf.math.exp(0.1 * (10 - epoch))


lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)


def train():
    train_data, val_data, train_weights, val_weights = prepare_data()
    train_ds = get_dataset(train_data)
    val_ds = get_dataset(val_data)
    model = unet()
    #model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=WeightedCCE(weights=train_weights),
                  metrics=['categorical_accuracy',
                           DiceCoeff1(),
                           DiceCoeff2()])

    tf.keras.utils.plot_model(model, model_dir + 'model.png')
    for idx, data in enumerate(train_ds):
        x = data[0]
        y = data[1]
        print("image shape: ", x.shape)
        print("label shape: ", y.shape)
        break

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                     factor=0.5,
                                                     patience=1,
                                                     min_lr=0.00001,
                                                     verbose=1)
    csv_logger = tf.keras.callbacks.CSVLogger(log_dir)
    model.fit(train_ds,
              epochs=10,
              validation_data=val_ds,
              callbacks=[early_stop, csv_logger])
    model.save(model_dir, save_format='tf')
    model.save(model_dir + "/model.hd5", save_format='h5')


if __name__ == "__main__":
    train()
