import tensorflow as tf
import os
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, BatchNormalization, ReLU, Input, ZeroPadding2D, Dropout, UpSampling2D, Concatenate, Cropping2D
from metric import *
from loss import *
from data_prep import *

## Adds dropout to conv layers + exp17
## patience changed from 10 to 5

BATCH_SIZE = 6
LR = 0.001
LOSS = 'WeightedCCE'
SPLITS = 5
EPOCHS = 1

exp = f"exp30_vgg2_lr{LR}_batch{BATCH_SIZE}_epochs{EPOCHS}_{LOSS}_ensemble{SPLITS}"
model_dir = "/home/rex/workspace/kaggle-severstal-defects/checkpoints/" + exp

log_dir = model_dir
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

    x = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    tf.print("End of each down-conv: ", x.shape)
    return x


def upconv_block(input, conv_op, filters):
    x1 = Conv2DTranspose(int(filters / 2), (2, 2), strides=(2, 2))(input)
    x = Concatenate()([x1, conv_op])

    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
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

    ## 112, 56, 28, 14, 7
    ## b5_pool= (7, 7, 512), last_conv = (7, 7, 1024) after
    upconv_block1 = upconv_block(last_conv, b5_c3, 512)

    upconv_block2 = upconv_block(upconv_block1, b4_c3, 512)  ## 28 after
    upconv_block3 = upconv_block(upconv_block2, b3_c3, 256)  ## 56
    upconv_block4 = upconv_block(upconv_block3, b2_c2, 128)  ## 112
    upconv_block5 = upconv_block(upconv_block4, b1_c3, 64)  ## 224

    # upconv_block4 = upconv_block(upconv_block4, down_conv.get_layer('block1_conv2').output, ((1, 1), (1, 1)),
    #                              64)  ## 224
    # fop1 = UpSampling2D()(upconv_block5)
    final_output = Conv2D(5, (1, 1), activation="softmax")(upconv_block5)

    print("final output shape: ", final_output.shape)

    # model output is (N, 128 * 800, 5) ## (N, 128 , 800, 5)
    model = tf.keras.Model(input, final_output)
    return model


def get_dataset(data):
    ds = tf.data.Dataset.from_tensor_slices(
        (data['image'].values, (data['1'].values, data['2'].values,
                                data['3'].values, data['4'].values)))
    ds = ds.map(lambda x, y: tf.py_function(read_dataset, [x, y],
                                            [tf.string, tf.float32]))
    ds = ds.map(lambda x, y: read_img(x, y))
    ds = ds.flat_map(lambda x, y: tile_dataset(x, y))
    ds = ds.shuffle(buffer_size=100, seed=40)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds


def scheduler(epoch):
    if epoch < 6:
        return 0.0001
    else:
        return 0.0001 * tf.math.exp(0.1 * (10 - epoch))


lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)


def train():
    _splits = prepare_data(num_splits=SPLITS)
    for idx, _split in enumerate(_splits):
        print(f"training split: {idx}")
        train_data, val_data, train_weights, val_weights = _split
        train_ds = get_dataset(train_data)
        val_ds = get_dataset(val_data)
        model = unet()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
            loss=getattr(loss, LOSS)(weights=train_weights),
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
        csv_logger = tf.keras.callbacks.CSVLogger(
            os.path.join(log_dir, f"training_{idx}.csv"))
        model.fit(train_ds,
                  epochs=EPOCHS,
                  validation_data=val_ds,
                  callbacks=[early_stop, csv_logger])
        model.save(model_dir, save_format='tf')
        model.save(model_dir + f"/{exp}_model{idx}.hd5", save_format='h5')


if __name__ == "__main__":
    train()
