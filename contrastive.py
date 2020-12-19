import tensorflow as tf
from tensorflow.keras import layers
import utils
import cv2
import os
import tf_utils
import data_utils
import functools
import constants
import data


class BilinearProduct(tf.keras.layers.Layer):
    """Bilinear product."""

    def __init__(self, dim):
        super().__init__()
        self._dim = dim

    def build(self, _):
        self._w = self.add_weight(
            shape=(self._dim, self._dim),
            initializer="random_normal",
            trainable=True,
            name="bilinear_product_weight",
        )

    def call(self, anchor, positive):
        projection_positive = tf.linalg.matmul(self._w, positive, transpose_b=True)
        return tf.linalg.matmul(anchor, projection_positive)


class ContrastiveModel(tf.keras.Model):
    def __init__(self, encoder, num_hidden_units=512):
        super().__init__()
        self.encoder = encoder
        self.projector = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(num_hidden_units, activation=None),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Activation(tf.keras.activations.tanh),
            ]
        )
        self.similarity = BilinearProduct(num_hidden_units)

    @tf.function
    def train_step(self, data):
        views, is_bonafide = data
        sparse_labels = tf.cast(
            tf.expand_dims(is_bonafide, axis=1) == tf.expand_dims(is_bonafide, axis=0),
            tf.float32,
        )
        with tf.GradientTape() as tape:
            x = self.encoder(views)
            x = self.projector(x)
            similarities = self.similarity(x, x)
            loss = self.compiled_loss(sparse_labels, similarities)
            loss += sum(self.losses)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(sparse_labels, similarities)
        return {m.name: m.result() for m in self.metrics}


def augment_pretrain_dataset(dataset):
    bf_dataset = dataset
    bf_dataset = bf_dataset.map(
        tf_utils.create_set_shape_fn([None], []),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    bf_dataset = bf_dataset.map(
        functools.partial(
            data.create_spectogram, n_frames=constants.N_FRAMES, is_training=True,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    bf_dataset = tf.data.Dataset.zip(
        (
            bf_dataset,
            bf_dataset.map(
                # tf_utils.create_oh_map_fn(y, depth),
                lambda *x: 1.0,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            ),
        )
    )

    sp_dataset = dataset.map(
        # tf.function(
        lambda x, y: tf.numpy_function(
            func=lambda a, b: (
                utils.apply_sequential_transforms(a, b, return_audio=True),
                b,
            ),
            inp=[x, y],
            Tout=(tf.float32, tf.int64),
        )
        # )
        ,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    sp_dataset = sp_dataset.map(
        tf_utils.create_set_shape_fn([None], []),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    sp_dataset = sp_dataset.map(
        functools.partial(
            data.create_spectogram, n_frames=constants.N_FRAMES, is_training=True,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    sp_dataset = tf.data.Dataset.zip(
        (
            sp_dataset,
            sp_dataset.map(
                # tf_utils.create_oh_map_fn(y, depth),
                lambda *x: 0.0,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            ),
        )
    )
    return tf.data.experimental.sample_from_datasets(
        [bf_dataset.repeat(), sp_dataset.repeat()]
    )


def pretrain(
    model,
    pretrain_dataset,
    exp_path="./ssl_runs",
    ckpt_path="./checkpoints/pt-cp-{epoch:04d}.ckpt",
    load_path=None,
    run_pretrain=True,
    tb_path="logs",
    hidden_dim=512,
    batch_size=64,
    epochs=25,
    learning_rate=0.001,
    steps_per_epoch=80,
):
    contrastive_model = ContrastiveModel(model, hidden_dim)
    contrastive_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.BinaryAccuracy()],
    )
    if load_path is not None:
        if not load_path.endswith("ckpt"):
            load_path = tf.train.latest_checkpoint(load_path)
        contrastive_model.load_weights(load_path)
    if not run_pretrain:
        return contrastive_model.encoder
    dataset = (
        augment_pretrain_dataset(pretrain_dataset.repeat().shuffle(buffer_size=1000))
        .batch(batch_size)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path, save_weights_only=True, save_freq=25 * steps_per_epoch
    )

    backup_path = os.path.join(exp_path, "backup")
    backandrestore_callback = tf.keras.callbacks.experimental.BackupAndRestore(
        backup_dir=backup_path
    )

    log_dir = os.path.join(exp_path, "logs")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    callbacks = [
        model_checkpoint_callback,
        backandrestore_callback,
        tensorboard_callback,
    ]
    contrastive_model.fit(
        dataset, epochs=epochs, callbacks=callbacks, steps_per_epoch=steps_per_epoch
    )
    return contrastive_model.encoder
