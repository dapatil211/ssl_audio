import tensorflow as tf
from tensorflow.keras import layers
import utils
import cv2
import os
import tf_utils
import data
import constants
import functools


class BYOLModel(tf.keras.Model):
    def __init__(
        self, encoder, num_hidden_units=1024, output_dim=256, l2_regularization=0.0001
    ):
        super().__init__()
        self.l2_regularization = l2_regularization
        self.encoder = encoder
        self.online_projector = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(num_hidden_units, activation=None),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dense(output_dim, activation=None),
            ]
        )
        self.online_predictor = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(num_hidden_units, activation=None),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dense(output_dim, activation=None),
            ]
        )
        self.target_encoder = tf.keras.models.clone_model(self.encoder)
        self.target_projector = tf.keras.models.clone_model(self.online_projector)
        self.target_encoder.trainable = False
        self.target_projector.trainable = False

    @tf.function
    def train_step(self, data):
        d1_views, d2_views = data
        with tf.GradientTape() as tape:
            d1_online, d2_target = self.forward_pass(d1_views, d2_views)
            d2_online, d1_target = self.forward_pass(d2_views, d1_views)
            loss = self.compiled_loss(d1_online, d2_target)
            loss += self.compiled_loss(d2_online, d1_target)
            trainable_vars = (
                self.encoder.trainable_variables
                + self.online_projector.trainable_variables
                + self.online_predictor.trainable_variables
            )
            l2_loss = (
                tf.reduce_sum(
                    [tf.norm(v) for v in trainable_vars if "kernel" in v.name]
                )
                * self.l2_regularization
            )
            loss += l2_loss
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.align_target_network()
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def forward_pass(self, online_views, target_views):
        y_online = self.encoder(online_views)
        z_online = self.online_projector(y_online)
        q_online = self.online_predictor(z_online)
        # q_online = tf.math.l2_normalize(q_online, axis=-1)
        y_target = self.target_encoder(target_views)
        z_target = self.target_projector(y_target)
        # z_target = tf.math.l2_normalize(z_target, axis=-1)
        return q_online, tf.stop_gradient(z_target)

    @tf.function
    def align_target_network(self, decay=0.996):
        for o, t in zip(self.encoder.variables, self.target_encoder.variables):
            t.assign(o * (1 - decay) + t * decay)
        for o, t in zip(
            self.online_projector.variables, self.target_projector.variables
        ):
            t.assign(o * (1 - decay) + t * decay)


class BYOLLoss(tf.keras.losses.Loss):
    @tf.function
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        return 2 - 2 * tf.reduce_sum(
            tf.math.l2_normalize(y_pred, axis=-1)
            * tf.math.l2_normalize(y_true, axis=-1),
            axis=-1,
        )


def augment_pretrain_dataset(dataset):
    dataset_anchors = dataset.map(
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
    dataset_positives = dataset.map(
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
    # dataset = dataset.map(
    #     # tf.function(
    #     lambda x, y: tf.numpy_function(
    #         func=lambda a, b: (
    #             (utils.apply_sequential_transforms(a, b, return_audio=True), b),
    #             (utils.apply_sequential_transforms(a, b, return_audio=True), b),
    #         ),
    #         inp=[x, y],
    #         Tout=(tf.float32, tf.float32),
    #     )
    #     # )
    #     ,
    #     num_parallel_calls=tf.data.experimental.AUTOTUNE,
    # )
    dataset_anchors = dataset_anchors.map(
        tf_utils.create_set_shape_fn([None], []),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    dataset_positives = dataset_positives.map(
        tf_utils.create_set_shape_fn([None], []),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    dataset_anchors = dataset_anchors.map(
        functools.partial(
            data.create_spectogram, n_frames=constants.N_FRAMES, is_training=True,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    dataset_positives = dataset_positives.map(
        functools.partial(
            data.create_spectogram, n_frames=constants.N_FRAMES, is_training=True,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    dataset = tf.data.Dataset.zip((dataset_anchors, dataset_positives))

    # rescale = tf.keras.Sequential(
    #     [layers.experimental.preprocessing.Rescaling(1.0 / 255)]
    # )

    # dataset = dataset.map(
    #     # tf.function(
    #     lambda x, y: (tf.expand_dims(x / 255, -1), tf.expand_dims(y / 255, -1))
    #     # )
    #     ,
    #     num_parallel_calls=tf.data.experimental.AUTOTUNE,
    # )
    # dataset = dataset.map(
    #     tf_utils.create_set_shape_fn((224, 224, 1), (224, 224, 1)),
    #     num_parallel_calls=tf.data.experimental.AUTOTUNE,
    # )
    # print(dataset)
    return dataset


def pretrain(
    model,
    pretrain_dataset,
    exp_path="./ssl_runs",
    ckpt_path="./checkpoints/pt-cp-{epoch:04d}.ckpt",
    load_path=None,
    run_pretrain=True,
    tb_path="logs",
    hidden_dim=1024,
    output_dim=256,
    batch_size=64,
    epochs=25,
    learning_rate=0.001,
    steps_per_epoch=40,
):
    byol_model = BYOLModel(model, hidden_dim, output_dim)
    byol_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate), loss=BYOLLoss()
    )
    if load_path is not None:
        if not load_path.endswith("ckpt"):
            load_path = tf.train.latest_checkpoint(load_path)
        byol_model.load_weights(load_path)
    if not run_pretrain:
        return byol_model.encoder
    dataset = (
        augment_pretrain_dataset(pretrain_dataset.repeat().shuffle(buffer_size=1000))
        .batch(batch_size)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    # latest = tf.train.latest_checkpoint(os.path.dirname(ckpt_path))
    # byol_model.load_weights(latest)
    # return

    # loss_fn = tf.function(
    #     lambda x, y: 2
    #     - 2
    #     * tf.reduce_sum(
    #         tf.math.l2_normalize(x, axis=-1) * tf.math.l2_normalize(x, axis=-1), axis=-1
    #     )
    # )
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
    # tf.config.run_functions_eagerly(True)
    # import ipdb

    # ipdb.set_trace()
    byol_model.fit(
        dataset, epochs=epochs, callbacks=callbacks, steps_per_epoch=steps_per_epoch
    )
    return byol_model.encoder

