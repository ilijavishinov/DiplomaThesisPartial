from typing import Dict
import itertools
import pandas as pd
from scikeras.wrappers import KerasClassifier
from keras_metrics_module import f1_1_func, f1_0_func, recall_func, precision_func, f1_macro_func
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, ReLU
import numpy as np
import tensorflow as tf


def create_hyperparam_grid(search_space: Dict, experiment_id: str, dir: str):
    # combinations
    keys, values = zip(*search_space.items())
    combinations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # tabular combinations
    model_names = [f'{experiment_id}_{i + 1}' for i in range(len(combinations_dicts))]
    combinations_df = pd.DataFrame(combinations_dicts)
    combinations_df.insert(0, 'Model', model_names)
    combinations_df['Trained'] = 'No'

    # save
    search_space_path = f'{dir}/search_space_{experiment_id}.xlsx'
    combinations_df.to_excel(search_space_path, index = False)
    
    return search_space_path


def build_model(
        input_shape,
        num_dense_layers,
        first_dense_layer_num_nodes,
        dense_layers_shrinkage_factor,
        loss_name,
        batch_norm,
        optimizer_name,
        learning_rate,
        dropout,
        l2_lambda
    ):

    layers_list = [Dense(first_dense_layer_num_nodes,
                         input_shape = input_shape,
                         kernel_regularizer = tf.keras.regularizers.l2(l2 = l2_lambda))]
    layers_list.append(ReLU())
    if batch_norm:
        layers_list.append(BatchNormalization())
    if dropout:
        layers_list.append(Dropout(0.1))

    dense_layer_current_num_nodes = first_dense_layer_num_nodes
    for dense_layer_number in range(1,num_dense_layers+1):
        dense_layer_current_num_nodes = dense_layer_current_num_nodes * dense_layers_shrinkage_factor
        layers_list.append(Dense(dense_layer_current_num_nodes,
                                 kernel_regularizer = tf.keras.regularizers.l2(l2 = l2_lambda)))
        layers_list.append(ReLU())
        if batch_norm:
            layers_list.append(BatchNormalization())
        if dropout:
            layers_list.append(Dropout(0.1))

    if dropout:
        layers_list = layers_list[:-1]
    layers_list.append(Dense(1, activation='sigmoid'))

    if optimizer_name.lower() == 'adam':
        optimizer = tf.optimizers.Adam(learning_rate = learning_rate)

    built_model = Sequential(layers_list)
    built_model.compile(loss=loss_name, optimizer=optimizer,
                        metrics=['accuracy', f1_macro_func, f1_1_func, f1_0_func, recall_func, precision_func])
    tf.keras.utils.plot_model(built_model, to_file='model.pdf', show_shapes=True)

    return built_model
