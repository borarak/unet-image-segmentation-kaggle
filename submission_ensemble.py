import os
import pandas as pd
import numpy as np
import time

sample_csv_path = "../input/severstal-steel-defect-detection/sample_submission.csv"
export_csv_path = "submission.csv"
test_image_dir = "../input/severstal-steel-defect-detection/test_images/"

model1_path = "../input/exp41-ensemble/exp41_vgg2_lr0001_batch32_epochs10_WeightedCCE_ensemble3_model0.hd5"
model = tf.keras.models.load_model(model1_path, compile=False)
model_cfg = model.get_config()
model1 = tf.keras.Model.from_config(model_cfg)
model1.set_weights(model.get_weights())
print("Model1 loaded....")

model2_path = "../input/exp41-ensemble/exp41_vgg2_lr0001_batch32_epochs10_WeightedCCE_ensemble3_model1.hd5"
model = tf.keras.models.load_model(model2_path, compile=False)
model_cfg = model.get_config()
model2 = tf.keras.Model.from_config(model_cfg)
model2.set_weights(model.get_weights())
print("Model2 loaded....")

model3_path = "../input/exp41-ensemble/exp41_vgg2_lr0001_batch32_epochs10_WeightedCCE_ensemble3_model2.hd5"
model = tf.keras.models.load_model(model3_path, compile=False)
model_cfg = model.get_config()
model3 = tf.keras.Model.from_config(model_cfg)
model3.set_weights(model.get_weights())
print("Model3 loaded....")


def mask2rle(img):
    '''
    img: numpy array, 1 -> mask, 0 -> background
    Returns run length as string formated
    '''
    img = img.reshape((256, 1600))
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


prediction_df = []
start_time = time.time()
for idx, image in enumerate(os.listdir(test_image_dir)):
    image_file = test_image_dir + image
    if image.endswith("jpg"):
        print("Processing image: {} - {}".format(idx, image))
        # Crop and make predictions on each crop

        x = tf.expand_dims(tf.image.decode_jpeg(tf.io.read_file(image_file)),
                           axis=0)

        x_crop_boxes = []
        prediction_boxes = []

        for _start_width in range(0, 1600, 200):
            cropped_image = tf.image.resize(
                tf.image.crop_to_bounding_box(x, 0, _start_width, 128, 200),
                (224, 224))
            _pred1 = model1.predict(cropped_image)
            _pred2 = model2.predict(cropped_image)
            # _pred3 = model3.predict(cropped_image)
            _pred = tf.reduce_max(tf.concat([_pred1, _pred2], axis=0),
                                  axis=0,
                                  keep_dims=True)
            # print("Single model crop prediction shape", _pred1.shape)
            # print("Ensemble prediction shape", _pred.shape)

            prediction_boxes.append(_pred)

        full_image_prediction = np.concatenate(prediction_boxes, axis=2)
        # print("full prediction size: ", full_image_prediction.shape)
        prediction = tf.convert_to_tensor(full_image_prediction)
        prediction = tf.image.resize(prediction, (256, 1600))
        print("full prediction size after resizing: ", prediction.shape)

        zeros_ = tf.zeros_like(prediction)
        ones_ = tf.ones_like(prediction)
        final_pred = tf.where(
            tf.equal(prediction,
                     tf.reduce_max(prediction, axis=-1, keepdims=True)), ones_,
            zeros_)

        if tf.reduce_sum(final_pred[:, :, :, 1:]).numpy() <= 80:
            _ones_ = tf.ones((1, 256, 1600, 1))
            _zeros_ = tf.zeros((1, 256, 1600, 1))
            final_pred = tf.concat(
                [_ones_, _zeros_, _zeros_, _zeros_, _zeros_], axis=-1)
        # print("Mother of all shapes: ", full_image_prediction.shape)

        prediction = final_pred.numpy()[0]
        print("prediction shape (mother of all shapes)", prediction.shape)

        c1 = prediction[:, :, 1]  # .reshape((-1, 1))
        c2 = prediction[:, :, 2]  # .reshape((-1, 1))
        c3 = prediction[:, :, 3]  # .reshape((-1, 1))
        c4 = prediction[:, :, 4]  # .reshape((-1, 1))

        # prediction_df.append({'ImageId_ClassId': image + "_0", 'EncodedPixels': mask2rle(c0)})
        prediction_df.append({
            'ImageId_ClassId': image + "_1",
            'EncodedPixels': mask2rle(c1)
        })
        prediction_df.append({
            'ImageId_ClassId': image + "_2",
            'EncodedPixels': mask2rle(c2)
        })
        prediction_df.append({
            'ImageId_ClassId': image + "_3",
            'EncodedPixels': mask2rle(c3)
        })
        prediction_df.append({
            'ImageId_ClassId': image + "_4",
            'EncodedPixels': mask2rle(c4)
        })

df = pd.DataFrame.from_dict(prediction_df)
df.to_csv(export_csv_path, index=False)

end_time = time.time()
print("done finally, total time taken...!", (end_time - start_time))
