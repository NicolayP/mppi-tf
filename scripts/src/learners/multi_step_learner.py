import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

class WindowGenerator():
    def __init__(self, inWidth, labelWidth, shift, batchSize,
                 train, val=None, test=None,
                 labelCols=None, actCols=None, stateCols=None, seq=False):
        # Data, df lists, one for each trajectory
        self.train_df = train
        self.val_df = val
        self.test_df = test

        self.batchSize = batchSize

        # Work out the label column indices.
        self.labelCols = labelCols
        if labelCols is not None:
            self.labelColIdx = {name: i for i, name in enumerate(labelCols)}

        self.seq = seq
        self.actCols = actCols
        self.stateCols = stateCols

        inCols = actCols + stateCols


        self.inCols = inCols
        if inCols is not None:
            self.inColIdx = {name: i for i, name in enumerate(inCols)}

        self.colIdx = {name: i for i, name in enumerate(train.columns)}

        # Work out the window parameters.
        self.inWidth = inWidth
        self.labelWidth = labelWidth
        self.shift = shift

        self.totalWinSize = inWidth + shift
        self.inSlice = slice(0, inWidth)
        self.inIdx = np.arange(self.totalWinSize)[self.inSlice]

        self.labelStart = self.totalWinSize - self.labelWidth
        self.labelSlice = slice(self.labelStart, None)
        self.labelIdx = np.arange(self.totalWinSize)[self.labelSlice]

    def split_window(self, features):
        inputs = features[:, self.inSlice, :]
        labels = features[:, self.labelSlice, :]

        if self.labelCols is not None:
            labels = tf.stack(
                [labels[:, :, self.colIdx[name]] for name in self.labelCols],
                axis=-1)

        if self.inCols is not None:
            inputs = tf.stack(
                [inputs[:, :, self.colIdx[name]] for name in self.inCols],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the 'tf.data.Dataset' are easier to inspect.
        #inputs.set_shape([None, self.inWidth, None])
        #labels.set_shape([None, self.labelWidth, None])
        return inputs, labels

    def sequence(self, inputs):
        # input shape : (batch, inWidth, sDim+aDim)
        # output shape : batch, 1, sdim + inWidth*aDim)
        state = tf.stack(
                [inputs[:, 0:, self.colIdx[name]] for name in self.stateCols],
                axis=-1)

        actionSeq = tf.stack(
                [inputs[:, :, self.colIdx[name]] for name in self.actCols],
                axis=-1)
        return state, actionSeq

    def split_window_seq(self, features):
        # This assumes that the model knows the action sequence in advance
        # need to set the window parameters as follow:
        # inWidth = Tau
        # label = Tau
        # shift = 1
        # Labels stays the same as previously.
        state, actionSeq = self.sequence(features[:, self.inSlice, :])
        labels = features[:, self.labelSlice, :]

        if self.labelCols is not None:
            labels = tf.stack(
                [labels[:, :, self.colIdx[name]] for name in self.labelCols],
                axis=-1)
        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the 'tf.data.Dataset' are easier to inspect.

        actionSeq.set_shape([None, self.inWidth, None])
        state.set_shape([None, self.inWidth, None])
        labels.set_shape([None, self.labelWidth, None])
        return state, actionSeq, labels

    def plot(self, features, model=None, plotCols=['foo']):
        '''
            plot features for a given sequence. If model is given, it also plots
            the prediciton
        '''
        if self.seq:
            states, actionSeq, labels = self.split_window_seq(features)
            inputs = (tf.expand_dims(states[0, 0, :], axis=0),
                      tf.expand_dims(actionSeq[0], axis=0))
            inputsPlt, _ = self.split_window(features)
        else:
            inputs, labels = self.split_window(features)

        plt.figure(figsize=(20, 16))
        plotColIdx = [self.colIdx[name] for name in plotCols]
        maxN = len(plotCols)

        if model is not None:
            predicitons = model(inputs)

        for i, name in enumerate(plotCols):
            plt.subplot(maxN, 1, i+1)
            plt.ylabel(f'{name} [normed]')

            if self.seq:
                plt.plot(self.inIdx, inputsPlt[0, :, self.inColIdx[name]],
                         label='Inputs', marker='.', zorder=-10)
            else:
                plt.plot(self.inIdx, inputs[0, :, self.inColIdx[name]],
                         label='Inputs', marker='.', zorder=-10)

            if self.labelCols:
                labelColIdx = self.labelColIdx[name]
            else:
                labelColIdx = plotColIdx

            if labelColIdx is None:
                continue

            plt.scatter(self.labelIdx, labels[0, :, labelColIdx],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)

            if model is not None:
                plt.scatter(self.labelIdx, predicitons[0, :, labelColIdx],
                            marker='X', edgecolors='k', label='Predicitons',
                            c='#ff7f0e', s=64)

            if i == 0:
                plt.legend()
        plt.xlabel('Time [h]')

    def filter_nan(self, x, y, z):
        return (not tf.reduce_any(tf.math.is_nan(x))\
                and not tf.reduce_any(tf.math.is_nan(y))\
                and not tf.reduce_any(tf.math.is_nan(z)))

    def make_dataset(self, frame):
        data = np.array(frame, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=data,
                targets=None,
                sequence_length=self.totalWinSize,
                sequence_stride=1,
                shuffle=True,
                batch_size=self.batchSize,)
        if self.seq:
            ds = ds.map(self.split_window_seq)
            ds = ds.filter(self.filter_nan)

        else:
            ds = ds.map(self.split_window)
            filterNan = lambda x, y : not tf.reduce_any(tf.math.is_nan(x)) \
                                      and not tf.reduce_any(tf.math.is_nan(y))
            ds = ds.filter(filterNan)

        return ds

    @property
    def train(self):
        if self.train_df is not None:
            return self.make_dataset(self.train_df)
        return self.train_df

    @property
    def val(self):
        if self.val_df is not None:
            return self.make_dataset(self.val_df)
        return self.val_df

    @property
    def test(self):
        if self.test_df is not None:
            return self.make_dataset(self.test_df)
        return self.test_df
    
    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.totalWinSize}',
            f'Input indices: {self.inIdx}',
            f'Label indices: {self.labelIdx}',
            f'Label column name(s): {self.labelCols}',
            f'Sequence prediction: {self.seq}'])
