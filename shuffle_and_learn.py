import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import utils
import tf_utils
import data
import constants
import functools
import os


class SALModel(tf.keras.Model):
    def __init__(self, encoder, num_hidden_units=1024, output_dim=256):
        super().__init__()
        self.encoder = encoder
        self.predictor = tf.keras.Sequential(
            [
            tf.keras.layers.Dense(num_hidden_units, activation=None),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(output_dim), # , activation='sigmoid'
            ]
        )

    @tf.function
    def train_step(self, data):
        batch, targets = data
        with tf.GradientTape() as tape:
            output = self.forward_pass(batch)
            loss = self.compiled_loss(targets, output)
        trainable_vars = (
            self.encoder.trainable_variables
            + self.predictor.trainable_variables
        )
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def forward_pass(self, x):
        out = self.encoder(x)
        out = self.predictor(out)
        return out

def shuffle_seq(x,y):
    # Positive Sequence:
    # a, b, c
    # Negative Sequences:
    # b, a, d
    # d, a, b
    a, b, c, d = np.array_split(x, 4) #np.array_split(x, 4) #,axis=1
    if np.random.randint(2) < .5: # 50% chance of positive and 50% negative
        seq = tf.concat([a, b, c], axis=0)
        y = tf.cast(1, tf.float32) # float32 int32
    elif np.random.randint(2) < .5: # 25% first type of negative
        seq = tf.concat([b, a, d], axis=0)
        y = tf.cast(0, tf.float32)
    else:  # 25% second type of negative
        seq = tf.concat([d, a, b], axis=0)
        y = tf.cast(0, tf.float32)
    return seq, [y]


def shuffle_pretext_task_dataset(dataset, use_augmentation):

    # create the shuffle pretext task
    dataset = dataset.map(
        lambda x, y: tf.numpy_function(
            func=lambda a,b: shuffle_seq(a,b),
            inp=[x,y],
            Tout=(tf.float32, tf.float32),
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    if use_augmentation:
        # Augmentations
        dataset = dataset.map(
            lambda x, y: tf.numpy_function(
                func=lambda a, b: (
                    utils.apply_sequential_transforms(a, b, return_audio=True),
                    b,
                ),
                inp=[x, y],
                Tout=(tf.float32, tf.int64),
            )
            ,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

    dataset = dataset.map(
        tf_utils.create_set_shape_fn([None], [1]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    dataset = dataset.map(
        functools.partial(
            data.create_spectogram_with_label, n_frames=constants.N_FRAMES, is_training=True,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
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
        output_dim=1,
        batch_size=64,
        epochs=25,
        learning_rate=0.001,
        steps_per_epoch=40,
):
    sal_model = SALModel(model, hidden_dim, output_dim)
    sal_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=["accuracy", "loss"])
    if load_path is not None:
        if not load_path.endswith("ckpt"):
            load_path = tf.train.latest_checkpoint(load_path)
        sal_model.load_weights(load_path)
    if not run_pretrain:
        return sal_model.encoder
    use_augmentation = False
    dataset = (
        shuffle_pretext_task_dataset(pretrain_dataset.repeat(), use_augmentation)
        .shuffle(buffer_size=1024)
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

    log_dir = os.path.join(tb_path, "sal_pretrain")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    callbacks = [
        model_checkpoint_callback,
        backandrestore_callback,
        tensorboard_callback,
    ]
    sal_model.fit(dataset, epochs=epochs, callbacks=callbacks, steps_per_epoch=steps_per_epoch)
    return sal_model.encoder
