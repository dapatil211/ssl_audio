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


class ColaModel(tf.keras.Model):
    """Wrapper class for custom contrastive model."""

    def __init__(self, embedding_model, similarity_layer):
        super().__init__()
        self.embedding_model = embedding_model
        self._similarity_layer = similarity_layer

    def train_step(self, data):
        anchors, positives = data

        with tf.GradientTape() as tape:
            inputs = tf.concat([anchors, positives], axis=0)
            embeddings = self.embedding_model(inputs, training=True)
            anchor_embeddings, positive_embeddings = tf.split(embeddings, 2, axis=0)

            # logits
            similarities = self._similarity_layer(
                anchor_embeddings, positive_embeddings
            )

            sparse_labels = tf.range(tf.shape(anchors)[0])

            loss = self.compiled_loss(sparse_labels, similarities)
            loss += sum(self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(sparse_labels, similarities)
        return {m.name: m.result() for m in self.metrics}


def augment_pretrain_dataset(dataset, noise=0.001):
    dataset = dataset.map(
        functools.partial(prepare_example, noise=noise),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    return dataset


def get_contrastive_network(encoder, embedding_dim=512, input_shape=(None, 64, 1)):
    """Creates a model for contrastive learning task."""
    inputs = tf.keras.layers.Input(input_shape)
    x = encoder(inputs)
    outputs = tf.keras.layers.Dense(embedding_dim, activation="linear")(x)
    outputs = tf.keras.layers.LayerNormalization()(outputs)
    outputs = tf.keras.layers.Activation("tanh")(outputs)
    embedding_model = tf.keras.Model(inputs, outputs)
    embedding_dim = embedding_model.output.shape[-1]
    similarity_layer = BilinearProduct(embedding_dim)
    return ColaModel(embedding_model, similarity_layer)


def prepare_example(example, sr, noise=0.001):
    """Creates an example (anchor-positive) for instance discrimination."""
    print(example)
    x = tf.math.l2_normalize(example, epsilon=1e-9)

    waveform_a = data.extract_window(x)
    mels_a = data.extract_log_mel_spectrogram(waveform_a)
    frames_anchors = mels_a[Ellipsis, tf.newaxis]

    waveform_p = data.extract_window(x)
    waveform_p = waveform_p + (noise * tf.random.normal(tf.shape(waveform_p)))
    mels_p = data.extract_log_mel_spectrogram(waveform_p)
    frames_positives = mels_p[Ellipsis, tf.newaxis]

    return frames_anchors, frames_positives


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
    noise=0.0,
    learning_rate=0.001,
    steps_per_epoch=40,
):
    cola_model = get_contrastive_network(model, hidden_dim)
    cola_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    if load_path is not None:
        if not load_path.endswith("ckpt"):
            load_path = tf.train.latest_checkpoint(load_path)
        cola_model.load_weights(load_path)

    if not run_pretrain:
        return cola_model.embedding_model.get_layer("backbone")
    dataset = (
        augment_pretrain_dataset(
            pretrain_dataset.repeat().shuffle(
                buffer_size=1000, reshuffle_each_iteration=True
            ),
            noise=noise,
        )
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
    cola_model.fit(
        dataset, epochs=epochs, callbacks=callbacks, steps_per_epoch=steps_per_epoch
    )
    return cola_model.embedding_model.get_layer("backbone")

