import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import random
import keras
import tensorflow as tf

from transformers import AutoTokenizer
from transformers import TFDistilBertModel, AutoConfig

import streamlit as st


class twitter_model:
    def __init__(self,model_weights="/kaggle/input/model-nlp-twitter/custom_model.keras"):
        #activate gpu
        gpu_devices = tf.config.experimental.list_physical_devices("GPU") 
        for device in gpu_devices: 
            tf.config.experimental.set_memory_growth(device, True)

        #define a tokenizer 
        self.tokenizer = AutoTokenizer.from_pretrained("sacculifer/dimbat_disaster_distilbert", do_lower_case=True)

        #define the pretrained model
        #model = TFAutoModelForSequenceClassification.from_pretrained("sacculifer/dimbat_disaster_distilbert")
        config = AutoConfig.from_pretrained('sacculifer/dimbat_disaster_distilbert')
        transformer = TFDistilBertModel.from_pretrained("sacculifer/dimbat_disaster_distilbert", config=config)

        input_ids = tf.keras.layers.Input(shape=(512,), name='input_ids', dtype='int32')
        attention_mask = tf.keras.layers.Input(shape=(512,), name='attention_mask', dtype='int32')

        embeddings = transformer(input_ids, attention_mask=attention_mask)[0]
        pooling = tf.keras.layers.GlobalAveragePooling1D()(embeddings)

        net = tf.keras.layers.BatchNormalization()(pooling)
        net = tf.keras.layers.Dense(1024, activation='relu')(net)
        net = tf.keras.layers.Dropout(0.2)(net)
        net = tf.keras.layers.Dense(1024, activation='relu')(net)
        net = tf.keras.layers.Dropout(0.2)(net)
        net = tf.keras.layers.Dense(1, activation='sigmoid')(net)

        self.model = tf.keras.Model(inputs=(input_ids, attention_mask), outputs=net)
        self.model.layers[2].trainable = True # freeze for transform layers

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=['accuracy']
        )

        # Loads the weights
        self.model.load_weights(model_weights)

    def predict(self, text_input="help there is an flood"):
        """token['input_ids']),token['attention_mask'])"""

        token= self.tokenizer(
                    text_input,
                    padding= "max_length",
                    add_special_tokens= True,
                    return_attention_mask= True,
                    return_token_type_ids= False
                )

        input_ids_tensor = tf.constant(token['input_ids'], dtype=tf.int32, shape=(1, 512))
        attention_mask_tensor = tf.constant(token['attention_mask'], dtype=tf.int32, shape=(1, 512))
        token_tensor={'input_ids': input_ids_tensor, 'attention_mask':attention_mask_tensor}
        prediction = self.model.predict(token_tensor)
        return prediction