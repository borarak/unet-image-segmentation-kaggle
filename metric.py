"""Custom metrics"""
import tensorflow as tf
from tensorflow.keras import backend as K


class DiceCoeff1(tf.keras.metrics.Metric):
    """Dice coeff without background clas"""
    def __init__(self, name='dice_coeff1', smooth=1, **kwargs):
        super(DiceCoeff1, self).__init__(name=name, **kwargs)
        self.smooth = smooth
        self.y_true = self.add_weight(name='y_true', initializer='zeros')
        self.y_pred = self.add_weight(name='y_pred', initializer='zeros')
        self.dice_coeff_val = self.add_weight(
            name='dice_coeff_value',
            initializer='zeros',
            aggregation=tf.compat.v1.VariableAggregation.MEAN)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """

        :param y_true: Y_true is Sparse (N, 409600, 1) -> [one label to each pixel]
        :param y_pred: Model outputs (N, 256 * 1600, 5) of logits
        :param sample_weight:
        :return:
        """
        dice_val = 0
        for i in range(1, 5):
            y_true_cls = K.flatten(y_true[:, :, i])
            y_pred_cls = K.flatten(y_pred[:, :, i])
            intersection = K.sum(y_true_cls * y_pred_cls)
            dice_val += (2. * intersection + self.smooth) / (
                K.sum(y_true_cls) + K.sum(y_pred_cls) + self.smooth)
        self.dice_coeff_val.assign(dice_val / 4)

    def result(self):
        return self.dice_coeff_val


class DiceCoeff2(tf.keras.metrics.Metric):
    """dice Coeff with background class included"""
    def __init__(self, name='dice_coeff2', smooth=1, **kwargs):
        super(DiceCoeff2, self).__init__(name=name, **kwargs)
        self.smooth = smooth
        self.y_true = self.add_weight(name='y_true', initializer='zeros')
        self.y_pred = self.add_weight(name='y_pred', initializer='zeros')
        self.dice_coeff_val = self.add_weight(
            name='dice_coeff_value',
            initializer='zeros',
            aggregation=tf.compat.v1.VariableAggregation.MEAN)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """

        :param y_true: Y_true is Sparse (N, 409600, 1) -> [one label to each pixel]
        :param y_pred: Model outputs (N, 256 * 1600, 5) of logits
        :param sample_weight:
        :return:
        """
        dice_val = 0
        for i in range(0, 5):
            y_true_cls = K.flatten(y_true[:, :, i])
            y_pred_cls = K.flatten(y_pred[:, :, i])
            intersection = K.sum(y_true_cls * y_pred_cls)
            dice_val += (2. * intersection + self.smooth) / (
                K.sum(y_true_cls) + K.sum(y_pred_cls) + self.smooth)
        self.dice_coeff_val.assign(dice_val / 5)

    def result(self):
        return self.dice_coeff_val
