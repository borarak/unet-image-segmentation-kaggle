"""Contains a few custom loss functions, competition used DiceLoss"""
import tensorflow as tf
from tensorflow.keras import backend as K


class WeightedCCE(tf.keras.losses.CategoricalCrossentropy):
    """Sampled Weighted Categorical Crossentropy"""
    def __init__(self, weights):
        super(WeightedCCE, self).__init__()
        self.weights = weights
        assert isinstance(self.weights, list)

    def __call__(self, y_true, y_pred):
        assert y_true.shape == y_pred.shape
        _all_weights = tf.convert_to_tensor(self.weights, dtype=tf.float32)

        init_weights = tf.zeros_like(y_true, dtype=tf.float32)
        positive_class = tf.ones_like(y_true, dtype=tf.float32)

        sample_weights = tf.where(tf.equal(y_true, positive_class),
                                  _all_weights, init_weights)
        sample_weights = tf.reduce_max(sample_weights, axis=-1)
        return super().__call__(y_true, y_pred, sample_weight=sample_weights)


class WeightedSCE(tf.keras.losses.SparseCategoricalCrossentropy):
    """Smaple Weighted Sparse categorical cross Entropy"""
    def __init__(self, weights):
        super(WeightedSCE, self).__init__()
        self.weights = weights
        assert isinstance(self.weights, list)

    def __call__(self, y_true, y_pred):
        assert len(y_true.shape) == 1  ## must be sparse
        assert len(self.weights) == y_true.shape[-1], "#Classes != #Weights"
        _all_weights = tf.convert_to_tensor(self.weights, dtype=tf.float32)

        y_true_float = tf.cast(y_true, dtype=tf.float32)
        y_true_hot = tf.one_hot(y_true,
                                axis=-1,
                                depth=len(self.weights),
                                dtype=tf.float32)

        init_weights = tf.zeros_like(y_true_hot, dtype=tf.float32)
        positive_class = tf.constant(1.0,
                                     shape=y_true_hot.shape,
                                     dtype=tf.float32)

        sample_weights = tf.where(tf.equal(y_true_hot, positive_class),
                                  _all_weights, init_weights)
        sample_weights = tf.reduce_max(sample_weights, axis=-1)
        return super().__call__(y_true, y_pred, sample_weight=sample_weights)


class DiceLoss(tf.keras.losses.Loss):
    def __init__(self,
                 name='dice_coeff',
                 reduction=tf.keras.losses.Reduction.AUTO,
                 smooth=1):
        super(DiceLoss, self).__init__(name=name, reduction=reduction)
        self.smooth = smooth

    def _single_dice_loss(self, y_true, y_pred):
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        intersection = K.sum(y_true * y_pred)
        return 1 - ((2. * intersection + self.smooth) /
                    (K.sum(y_true) + K.sum(y_pred) + self.smooth))

    def call(self, y_true, y_pred, num_labels=4):
        y_pred = tf.where(
            tf.equal(y_pred, tf.reduce_max(y_pred, axis=-1, keepdims=True)),
            tf.ones_like(y_pred), tf.zeros_like(y_pred))
        y_true = tf.reshape(
            tf.one_hot(tf.cast(y_true, dtype=tf.int32), depth=5, axis=-1),
            (-1, 256 * 1600, 5))
        loss = 0
        for i in range(0, 5):
            loss += self._single_dice_loss(y_true[:, :, i], y_pred[:, :, i])
        return loss / 5.0
