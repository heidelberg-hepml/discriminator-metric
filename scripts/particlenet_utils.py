import numpy as np
import awkward0
from tensorflow import keras
import wandb

def stack_arrays(a, keys, axis=-1):
    flat_arr = np.stack([a[k].flatten() for k in keys], axis=axis)
    return awkward0.JaggedArray.fromcounts(a[keys[0]].counts, flat_arr)

def pad_array(a, maxlen, value=0., dtype='float32'):
    x = (np.ones((len(a), maxlen)) * value).astype(dtype)
    for idx, s in enumerate(a):
        if not len(s):
            continue
        trunc = s[:maxlen].astype(dtype)
        x[idx, :len(trunc)] = trunc
    return x

import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

class Dataset(object):

    def __init__(self, filepath, feature_dict = {}, label='label', pad_len=100, data_format='channel_first'):
        self.filepath = filepath
        self.feature_dict = feature_dict
        if len(feature_dict)==0:
            feature_dict['points'] = ['part_etarel', 'part_phirel']
            feature_dict['features'] = ['part_pt_log', 'part_e_log', 'part_etarel', 'part_phirel']
            feature_dict['mask'] = ['part_pt_log']
        self.label = label
        self.pad_len = pad_len
        assert data_format in ('channel_first', 'channel_last')
        self.stack_axis = 1 if data_format=='channel_first' else -1
        self._values = {}
        self._label = None
        self._load()

    def _load(self):
        logging.info('Start loading file %s' % self.filepath)
        counts = None
        with awkward0.load(self.filepath) as a:
            self._label = a[self.label].reshape(-1,2)
            for k in self.feature_dict:
                cols = self.feature_dict[k]
                if not isinstance(cols, (list, tuple)):
                    cols = [cols]
                arrs = []
                for col in cols:
                    if counts is None:
                        counts = a[col].counts
                    else:
                        assert np.array_equal(counts, a[col].counts)
                    arrs.append(pad_array(a[col], self.pad_len))
                self._values[k] = np.stack(arrs, axis=self.stack_axis)
        logging.info('Finished loading file %s' % self.filepath)


    def __len__(self):
        return len(self._label)

    def __getitem__(self, key):
        if key==self.label:
            return self._label
        else:
            return self._values[key]
    
    @property
    def X(self):
        return self._values
    
    @property
    def y(self):
        return self._label

    def shuffle(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        shuffle_indices = np.arange(self.__len__())
        np.random.shuffle(shuffle_indices)
        for k in self._values:
            self._values[k] = self._values[k][shuffle_indices]
        self._label = self._label[shuffle_indices]

def lr_schedule(epoch):
    initial_lr = 3 * 1e-4
    lr = initial_lr
    if epoch <= 8:
        lr = initial_lr + (3 * 1e-3 - initial_lr) * epoch / 8
    elif (epoch <= 16) & (epoch > 8):
        lr = 3 * 1e-3 - (3 * 1e-3 - 3 * 1e-4) * (epoch - 8) / 8
    elif epoch > 16:
        lr = 3 * 1e-4 - (3 * 1e-4 - 5 * 1e-7) * (epoch - 16) / 4   
    return lr

class epoch_save(keras.callbacks.Callback):
    def __init__(self, model_path):
        self.model_path = model_path
    def on_epoch_end(self, epoch, logs=None):
        self.model.save(self.model_path.format(epoch=epoch))
        wandb.log({'loss': logs['loss'], 'val_loss': logs['val_loss'],
              'acc': logs['accuracy'], 'val_acc': logs['val_accuracy'],
            'learning_rate': self.model.optimizer.lr})