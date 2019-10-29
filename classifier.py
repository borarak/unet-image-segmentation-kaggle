import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from data_prep import prepare_data_classifier, read_img, tile_dataset, read_dataset

BATCH_SIZE = 64
LR = 0.0001
exp = f"exp01_classifier_lr{str(LR).split('.')[1]}_batch{BATCH_SIZE}"

base_dir = "/home/rex/workspace/kaggle-severstal-defects/checkpoints/"
model_dir = base_dir + exp

import os
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)


def make_classifier_label(x, y):
    y = tf.reduce_max(y[:, :, 1:])
    return x, y


def _hot_encode(x, y):
    print("before: ", y.shape)
    y = tf.one_hot(tf.cast(y, dtype=tf.int32),
                   depth=2,
                   axis=-1,
                   dtype=tf.float32)
    print("after: ", y.shape)
    return x, y


def get_dataset(data):
    ds = tf.data.Dataset.from_tensor_slices(
        (data['image'].values, (data['1'].values, data['2'].values,
                                data['3'].values, data['4'].values)))
    ds = ds.map(lambda x, y: tf.py_function(read_dataset, [x, y],
                                            [tf.string, tf.float32]))
    ds = ds.map(lambda x, y: read_img(x, y))
    ds = ds.flat_map(lambda x, y: tile_dataset(x, y))
    #
    ds = ds.map(lambda x, y: make_classifier_label(x, y))
    ds = ds.map(lambda x, y: _hot_encode(x, y))
    ds = ds.shuffle(buffer_size=100, seed=40)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds


import os
csv_logger = tf.keras.callbacks.CSVLogger(
    os.path.join(base_dir, f"training_classifier.csv"))
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

vgg = VGG16(weights='imagenet', include_top=False)
for layer in vgg.layers:
    trainable = False

input = tf.keras.layers.Input(shape=(224, 224, 3))
vgg_op = vgg(input)
flattens1 = tf.keras.layers.Flatten()(vgg_op)
dense1 = tf.keras.layers.Dense(4096)(flattens1)
drop1 = tf.keras.layers.Dropout(0.2)(dense1)
dense2 = tf.keras.layers.Dense(1000)(drop1)
drop2 = tf.keras.layers.Dropout(0.2)(dense2)
dense3 = tf.keras.layers.Dense(2, activation='softmax')(drop2)

model = tf.keras.models.Model(input, dense3)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
              loss='categorical_crossentropy',
              metrics=[
                  'categorical_accuracy',
                  tf.keras.metrics.TruePositives(),
                  tf.keras.metrics.FalsePositives(),
                  tf.keras.metrics.TrueNegatives(),
                  tf.keras.metrics.FalseNegatives()
              ])

train_data_final, val_data_final, train_weights, val_weights = prepare_data_classifier(
)
train_ds = get_dataset(train_data_final)
val_ds = get_dataset(val_data_final)

model.fit(train_ds,
          epochs=1,
          validation_data=val_ds,
          class_weight={
              0: 0.1,
              1: 0.9
          },
          callbacks=[early_stop, csv_logger])

model.save(model_dir + f"/{exp}_model.hd5", save_format='h5')
