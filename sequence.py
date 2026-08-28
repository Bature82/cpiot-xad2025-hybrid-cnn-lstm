"""Constant-memory batch generator (the only place windows become arrays)."""

import math

import numpy as np
import tensorflow as tf

from .windowing import target_offset


class WindowSeq(tf.keras.utils.Sequence):
    """Streams (batch, W, F) windows out of the single scaled 2-D array using
    start indices. Nothing larger than one batch is ever materialised, so RAM
    stays flat no matter how many windows exist."""

    def __init__(self, data2d, starts, w, n_net, mode, batch_size,
                 with_targets=True, shuffle=False, seed=0):
        try:
            super().__init__()           # Keras 3 (PyDataset)
        except TypeError:
            pass                         # Keras 2 Sequence
        self.d = data2d
        self.starts = np.asarray(starts, dtype=np.int64)
        self.w, self.n_net, self.mode = w, n_net, mode
        self.off = target_offset(w)
        self.bs = int(batch_size)
        self.with_targets = with_targets
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.order = np.arange(len(self.starts))
        if shuffle:
            self.rng.shuffle(self.order)
        # zero-copy VIEW of all possible windows: (N-w+1, F, w). One fancy
        # gather per batch replaces a per-sample python loop.
        self.swv = np.lib.stride_tricks.sliding_window_view(data2d, w, axis=0)

    def __len__(self):
        return int(math.ceil(len(self.starts) / self.bs))

    def __getitem__(self, i):
        idx = self.starts[self.order[i * self.bs:(i + 1) * self.bs]]
        Xb = self.swv[idx].transpose(0, 2, 1)   # (b, W, F): copy of THIS batch only
        tb = self.d[idx + self.off]             # (b, F): prediction target
        n = self.n_net
        if self.mode == "fused":
            # Named dicts, not tuples: Keras unpacks a bare tuple yielded by a
            # data iterator as (x, y), so a 2-input tuple in predict mode would
            # arrive as a single input tensor. A dict is always treated as one
            # multi-input x.
            ins = {"network_input": Xb[:, :, :n], "sensor_input": Xb[:, :, n:]}
            tgt = {"network_pred": tb[:, :n], "sensor_pred": tb[:, n:]}
        elif self.mode == "net_only":
            ins, tgt = Xb[:, :, :n], tb[:, :n]
        else:
            ins, tgt = Xb[:, :, n:], tb[:, n:]
        return (ins, tgt) if self.with_targets else ins

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.order)
