#!/usr/bin/env python
# coding: utf-8

# In[64]:


import os
import numpy as np
import random
import data_utils
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import datetime as dt
import argparse
import cv2
from tensorflow.keras.applications.vgg16 import VGG16
import tf_utils
import sys

sys.modules["keras"] = keras

# In[2]:


import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# In[52]:


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


# In[62]:


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain-data", default="./data/train/")
    parser.add_argument("--downstream-data-train", default="./data/train")
    parser.add_argument("--downstream-data-val", default="./data/dev")
    parser.add_argument("--downstream-data-test", default="./data/eval")
    parser.add_argument(
        "--pretrain-method",
        default="none",
        choices=["none", "byol", "contrastive", "cola", "sal"],
    )
    parser.add_argument("--dataset-creator", default="base")
    parser.add_argument("--experiment-dir", default="ssl_runs")
    parser.add_argument("--load-path", default=None)
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--ds-epochs", default=25, type=int)
    parser.add_argument("--pt-epochs", default=25, type=int)
    parser.add_argument("--data-size", default=45600, type=int)
    parser.add_argument("--init-lr", default=0.0001, type=float)
    parser.add_argument("--cola-noise", default=0.0, type=float)
    parser.add_argument("--custom-ds-model", action="store_true")

    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--load-spectogram", action="store_true")

    args = parser.parse_args()
    args.checkpoint_dir = os.path.join(
        args.experiment_dir, "checkpoints", r"cp-{epoch:04d}.ckpt"
    )
    args.tensorboard_dir = os.path.join(args.experiment_dir, "tb_logs")
    set_seed(args.seed)
    args.tensorboard_dir = args.tensorboard_dir + "/{}".format(
        dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    )
    dataset_creator = get_dataset_creator(args.dataset_creator, args)
    model = create_backbone_model(None, 64)
    run_pipeline(dataset_creator, model, args)


# In[6]:


def run_pipeline(dataset_creator, model, args):
    # Run self supervised training
    pretrain_dataset = dataset_creator.create_pt_dataset()
    model = run_pretrain(model, pretrain_dataset, args.pretrain_method, args)

    # Run downstream training
    (
        ds_train_data,
        ds_train_eval_data,
        ds_dev_data,
        ds_eval_data,
    ) = dataset_creator.create_ds_dataset()
    # print(model.summary())
    logger.info("FROZEN TRAINING AND EVALUATION")
    model_ds = create_downstream_model(model, freeze_backbone=True, custom_ds_model=args.custom_ds_model)
    train_downstream(
        model_ds,
        ds_train_data,
        ds_dev_data,
        args,
        exp_folder=os.path.join(args.experiment_dir, "frozen"),
    )
    print(model_ds.summary())
    # Run evaluation
    latest = tf.train.latest_checkpoint(
        os.path.join(args.experiment_dir, "frozen", "checkpoints")
    )
    model_ds.load_weights(latest)
    eval_downstream(
        model_ds,
        {"train_eval": ds_train_eval_data, "val": ds_dev_data, "test": ds_eval_data},
        args,
        exp_folder=os.path.join(args.experiment_dir, "frozen"),
    )

    logger.info("FINETUNE TRAINING AND EVALUATION")
    model_ds = create_downstream_model(model, freeze_backbone=False, custom_ds_model=args.custom_ds_model)
    train_downstream(
        model_ds,
        ds_train_data,
        ds_dev_data,
        args,
        exp_folder=os.path.join(args.experiment_dir, "finetune"),
    )
    print(model_ds.summary())
    latest = tf.train.latest_checkpoint(
        os.path.join(args.experiment_dir, "finetune", "checkpoints")
    )
    model_ds.load_weights(latest)
    # Run evaluation
    eval_downstream(
        model_ds,
        {"train_eval": ds_train_eval_data, "val": ds_dev_data, "test": ds_eval_data},
        args,
        exp_folder=os.path.join(args.experiment_dir, "finetune"),
    )


# In[73]:


def get_dataset_creator(dataset_creator, args):
    if dataset_creator == "base":
        from dataset import DatasetCreator

        dataset_creator = DatasetCreator(
            args.pretrain_data,
            args.downstream_data_train,
            args.downstream_data_val,
            args.downstream_data_test,
            args.load_spectogram,
        )
        return dataset_creator


# In[54]:


def create_backbone_model(img_width, img_height):
    inputs = keras.Input(shape=(img_width, img_height, 1))
    encoder = VGG16(
        input_tensor=inputs,
        weights=None,
        include_top=False,
        input_shape=(img_width, img_height, 1),
        pooling="max",
    )
    x = encoder(inputs)
    model = keras.Model(inputs=inputs, outputs=x, name="backbone")
    return model


# In[69]:


def create_downstream_model(backbone, n_bands=64, n_channels=1, freeze_backbone=False, custom_ds_model=False):
    inputs = tf.keras.layers.Input(shape=(None, n_bands, n_channels))
    x = backbone(inputs)
    backbone.trainable = not freeze_backbone
    if not custom_ds_model:
        predictions = layers.Dense(2, activation=None)(x)
    else:    
        x = layers.Conv2D(64, 3, activation=None)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.GlobalAveragePooling2D()(x)

        x = layers.Dense(256, activation=None)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        predictions = layers.Dense(2, activation=None)(x)

    # this is the model we will train
    model = keras.Model(inputs=inputs, outputs=predictions)
    return DSModel(model, n_bands=n_bands)


# In[53]:


# Import and run different pretrain method based on name
def run_pretrain(model, pretrain_dataset, pretrain_method, args):
    if pretrain_method == "none":
        return model

    elif pretrain_method == "byol":
        import byol

        return byol.pretrain(
            model,
            pretrain_dataset,
            exp_path=os.path.join(args.experiment_dir, "pretrain"),
            ckpt_path=os.path.join(
                args.experiment_dir, "pretrain", "checkpoints", r"cp-{epoch:04d}"
            ),
            load_path=args.load_path,
            run_pretrain=not args.skip_pretrain,
            tb_path=args.tensorboard_dir,
            epochs=args.pt_epochs,
        )
    elif pretrain_method == "contrastive":
        import contrastive

        return contrastive.pretrain(
            model,
            pretrain_dataset,
            exp_path=os.path.join(args.experiment_dir, "pretrain"),
            ckpt_path=os.path.join(
                args.experiment_dir, "pretrain", "checkpoints", r"cp-{epoch:04d}"
            ),
            load_path=args.load_path,
            run_pretrain=not args.skip_pretrain,
            tb_path=args.tensorboard_dir,
            epochs=args.pt_epochs,
        )
    elif pretrain_method == "cola":
        import cola

        return cola.pretrain(
            model,
            pretrain_dataset,
            exp_path=os.path.join(args.experiment_dir, "pretrain"),
            ckpt_path=os.path.join(
                args.experiment_dir, "pretrain", "checkpoints", r"cp-{epoch:04d}"
            ),
            load_path=args.load_path,
            run_pretrain=not args.skip_pretrain,
            tb_path=args.tensorboard_dir,
            epochs=args.pt_epochs,
            noise=args.cola_noise,
        )
    
    elif pretrain_method == "sal":
        import shuffle_and_learn

        return shuffle_and_learn.pretrain(
            model,
            pretrain_dataset,
            exp_path=os.path.join(args.experiment_dir, "pretrain"),
            ckpt_path=os.path.join(
                args.experiment_dir, "pretrain", "checkpoints", r"cp-{epoch:04d}"
            ),
            load_path=args.load_path,
            run_pretrain=not args.skip_pretrain,
            tb_path=args.tensorboard_dir,
            epochs=args.pt_epochs,
        )


# In[57]:
class DSModel(tf.keras.Model):
    def __init__(self, model, n_bands=64):
        super().__init__()
        self.model = model
        self.tdm = tf.keras.layers.TimeDistributed(model)

    def call(self, inputs, training=False):
        if training:
            x = self.model(inputs)
            return x
        else:
            x = self.tdm(inputs)
            x = tf.reduce_mean(x, axis=1)
            return x


def train_downstream(model, ds_train_data, ds_eval_data, args, exp_folder):
    ds_train_data = (
        ds_train_data.shuffle(buffer_size=512)
        .batch(args.batch_size)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    ds_eval_data = ds_eval_data.batch(1).prefetch(tf.data.experimental.AUTOTUNE)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.init_lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(), tf_utils.EERMetric()],
    )

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(exp_folder, "checkpoints", "cp.ckpt"),
        save_weights_only=True,
        verbose=1,
        monitor="val_eer",
        save_best_only=True,
        mode="min",
    )
    callbacks = [
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(exp_folder, "logs"), write_images=True
        ),
        cp_callback,
    ]
    history = model.fit(
        x=ds_train_data,
        epochs=args.ds_epochs,
        validation_data=ds_eval_data,
        callbacks=callbacks,
        #validation_freq=5,
        steps_per_epoch=args.data_size // args.batch_size,
    )
    return history


# In[58]:


def eval_downstream(model, ds_datasets, args, exp_folder, n_bands=64):

    for dataset_name in ds_datasets:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(args.init_lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                tf.keras.metrics.SparseCategoricalAccuracy(),
                tf_utils.EERMetric(),
            ],
        )
        dataset = (
            ds_datasets[dataset_name].batch(1).prefetch(tf.data.experimental.AUTOTUNE)
        )
        callbacks = [
            keras.callbacks.TensorBoard(
                log_dir=os.path.join(exp_folder, "logs", f"{dataset_name}_eval"),
                write_images=True,
            ),
        ]
        results = model.evaluate(dataset, return_dict=True, callbacks=callbacks)
        logger.info("%s results: %s", dataset_name, json.dumps(results))


# In[ ]:


if __name__ == "__main__":
    main()

