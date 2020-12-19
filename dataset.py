import os
import tensorflow as tf
import cv2
import data_utils
import numpy as np
from tensorflow.keras import layers
import tf_utils
import constants
import data
import functools


class DatasetCreator:
    def __init__(
        self,
        pt_data_folder,
        ds_train,
        ds_val,
        ds_test,
        load_spectogram_for_pt=False,
        n_frames=98,
        batch_size=64,
    ):
        self.pt_data_folder = pt_data_folder
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.ds_test = ds_test
        self.load_spectogram_for_pt = load_spectogram_for_pt
        self.n_frames = n_frames
        self.batch_size = 64

    def create_pt_dataset(self):
        return self.load_ds_dataset(
            os.path.join(self.pt_data_folder, "bonafide", "*"),
            load_spectogram=self.load_spectogram_for_pt,
        )

    def create_ds_dataset(self):
        train_ds_bf = self.load_ds_dataset(
            os.path.join(self.ds_train, "bonafide", "*"), is_training=True, y=1, depth=2
        )
        train_ds_sp = self.load_ds_dataset(
            os.path.join(self.ds_train, "spoof", "*"), is_training=True, y=0, depth=2
        )
        train_ds = tf.data.experimental.sample_from_datasets(
            [train_ds_bf.repeat(), train_ds_sp.repeat()]
        )

        train_eval_ds_bf = self.load_ds_dataset(
            os.path.join(self.ds_train, "bonafide", "*"), y=1, depth=2
        )
        train_eval_ds_sp = self.load_ds_dataset(
            os.path.join(self.ds_train, "spoof", "*"), y=0, depth=2
        )
        train_eval_ds = train_eval_ds_bf.concatenate(train_eval_ds_sp)

        val_ds_bf = self.load_ds_dataset(
            os.path.join(self.ds_val, "bonafide", "*"), y=1.0, depth=2
        )
        val_ds_sp = self.load_ds_dataset(
            os.path.join(self.ds_val, "spoof", "*"), y=0.0, depth=2
        )
        val_ds = val_ds_bf.concatenate(val_ds_sp)

        test_ds_bf = self.load_ds_dataset(
            os.path.join(self.ds_test, "bonafide", "*"), y=1.0, depth=2
        )
        test_ds_sp = self.load_ds_dataset(
            os.path.join(self.ds_test, "spoof", "*"), y=0.0, depth=2
        )
        test_ds = test_ds_bf.concatenate(test_ds_sp)

        return (
            train_ds,
            train_eval_ds,
            val_ds,
            test_ds,
            # train_ds_bf.concatenate(train_ds_sp),
            # train_ds_bf.concatenate(train_ds_sp),
            # train_ds_bf.concatenate(train_ds_sp),
            # val_ds_bf.concatenate(val_ds_sp),
            # test_ds_bf.concatenate(test_ds_sp),
        )

    def load_ds_dataset(
        self, folder, is_training=False, load_spectogram=True, y=None, depth=None
    ):
        # Confirm this pipeline, other code seems off
        # rescale = tf.keras.Sequential(
        #     [layers.experimental.preprocessing.Rescaling(1.0 / 255)]
        # )

        # Get the files
        files_ds = tf.data.Dataset.list_files(folder)

        # Load waveforms
        dataset = files_ds.map(
            lambda x: tf.numpy_function(
                func=lambda y: data_utils.file_load(y.decode("utf-8")),
                inp=[x],
                Tout=(tf.float32, tf.int64),
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        dataset = dataset.apply(tf.data.experimental.ignore_errors())
        dataset = dataset.map(
            tf_utils.create_set_shape_fn([None], []),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        # dataset = dataset.map(
        #     lambda audio: tf.cast(audio, tf.float32) / float(tf.int16.max)
        # )

        if load_spectogram:
            dataset = dataset.map(
                functools.partial(
                    data.create_spectogram,
                    n_frames=self.n_frames,
                    is_training=is_training,
                ),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
        #     # Convert to spectographs
        #     dataset = dataset.map(
        #         # tf.function(
        #         lambda x, _: tf.numpy_function(
        #             func=lambda y: cv2.resize(data_utils.to_sp(y), (224, 224)),
        #             inp=[x],
        #             Tout=tf.float32,
        #         )
        #         # )
        #         ,
        #         num_parallel_calls=tf.data.experimental.AUTOTUNE,
        #     )

        #     dataset = dataset.map(
        #         tf_utils.create_set_shape_fn((224, 224)),
        #         num_parallel_calls=tf.data.experimental.AUTOTUNE,
        #     )

        #     dataset = dataset.map(
        #         # tf.function(
        #         lambda x: tf.expand_dims(x / 255, -1)
        #         # )
        #         ,
        #         num_parallel_calls=tf.data.experimental.AUTOTUNE,
        #     )

        if y is not None:
            dataset = tf.data.Dataset.zip(
                (
                    dataset,
                    dataset.map(
                        # tf_utils.create_oh_map_fn(y, depth),
                        lambda *x: y,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE,
                    ),
                )
            )
        # parent_folder = os.path.join("/", *folder.split("/")[:-1])
        return dataset.cache()
        # os.path.abspath(
        #     os.path.join(
        #         parent_folder, "..", parent_folder.split("/")[-1] + ".cache"
        #     )
        # )

