import tensorflow as tf

from scipy.optimize import brentq
from scipy.interpolate import interp1d
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.ops import init_ops

from tensorflow.python.keras import backend as K


def create_set_shape_fn(*shapes):
    if isinstance(shapes[0], int):
        shapes = [shapes]

    # @tf.function
    def set_shape(*x):
        for i, t in enumerate(x):
            t.set_shape(shapes[i])
        if len(x) == 1:
            x = x[0]
        return x

    return set_shape


def create_oh_map_fn(y, depth):
    # @tf.function
    def oh_map_fn(*inp):
        return tf.one_hot(y, depth)

    return oh_map_fn


class EERMetric(tf.keras.metrics.Metric):
    def __init__(self, num_thresholds=1000, name="eer", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_thresholds = num_thresholds
        delta = 1 / self.num_thresholds
        self.thresholds = tf.expand_dims(
            tf.constant([delta * i for i in range(self.num_thresholds)]), axis=0
        )
        self.fp = self.add_weight(
            "fp", shape=(self.num_thresholds,), initializer=init_ops.zeros_initializer
        )
        self.tp = self.add_weight(
            "tp", shape=(self.num_thresholds,), initializer=init_ops.zeros_initializer
        )
        self.fn = self.add_weight(
            "fn", shape=(self.num_thresholds,), initializer=init_ops.zeros_initializer
        )
        self.tn = self.add_weight(
            "tn", shape=(self.num_thresholds,), initializer=init_ops.zeros_initializer
        )
        self.eer = self.add_weight("eer", initializer=init_ops.zeros_initializer)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.nn.softmax(y_pred, axis=1)
        scores = y_pred[:, 1:2]
        pred_labels = scores > self.thresholds
        self.tp.assign_add(
            tf.reduce_sum(
                tf.cast(
                    tf.logical_and(
                        pred_labels == tf.cast(y_true, tf.bool), y_true == 1
                    ),
                    tf.float32,
                ),
                axis=0,
            )
        )

        self.fp.assign_add(
            tf.reduce_sum(
                tf.cast(
                    tf.logical_and(
                        pred_labels != tf.cast(y_true, tf.bool), y_true == 0
                    ),
                    tf.float32,
                ),
                axis=0,
            )
        )

        self.tn.assign_add(
            tf.reduce_sum(
                tf.cast(
                    tf.logical_and(
                        pred_labels == tf.cast(y_true, tf.bool), y_true == 0
                    ),
                    tf.float32,
                ),
                axis=0,
            )
        )

        self.fn.assign_add(
            tf.reduce_sum(
                tf.cast(
                    tf.logical_and(
                        pred_labels != tf.cast(y_true, tf.bool), y_true == 1
                    ),
                    tf.float32,
                ),
                axis=0,
            )
        )
        fpr = self.fp / (self.fp + self.tn + 1e-12)
        fnr = self.fn / (self.tp + self.fn + 1e-12)
        threshold_id = tf.argmin(tf.abs(fpr - fnr))
        self.eer.assign((fpr[threshold_id] + fnr[threshold_id]) / 2)

    def result(self):
        return self.eer

    def reset_states(self):
        self.fp.assign(tf.zeros(self.num_thresholds))
        self.tp.assign(tf.zeros(self.num_thresholds))
        self.fn.assign(tf.zeros(self.num_thresholds))
        self.tn.assign(tf.zeros(self.num_thresholds))
        self.eer.assign(0.0)


@tf.function(
    input_signature=[tf.TensorSpec(None, tf.float32), tf.TensorSpec(None, tf.float32)]
)
def solve_for_eer(fpr, tpr):
    eer = tf.numpy_function(
        lambda a, b: brentq(lambda x: 1.0 - x - interp1d(a, b)(x), 0.0, 1.0),
        [fpr, tpr],
        tf.float32,
    )
    return eer


class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir, **kwargs):
        super().__init__(log_dir, **kwargs)
        self.lr_writer = tf.summary.create_file_writer(self.log_dir)

    def on_epoch_end(self, epoch, logs=None):
        lr = getattr(self.model.optimizer, "lr", None)
        with self.lr_writer.as_default():
            summary = tf.summary.scalar("learning_rate", lr, epoch)

        super().on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super().on_train_end(logs)
        self.lr_writer.close()


class LearningRateLoggingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        tf.summary.scalar("learning rate", data=lr, step=epoch)

