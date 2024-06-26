"""
Common utilities used across all experiments

author: William Tong (wtong@g.harvard.edu)
"""

from dataclasses import dataclass, field
import itertools
from pathlib import Path
import shutil
from typing import Callable, Iterable

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import sys
sys.path.append('../')
from train import train, train_step

def set_theme():
    sns.set_theme(style='ticks', font_scale=1.25, rc={
        'axes.spines.right': False,
        'axes.spines.top': False,
        'figure.figsize': (4, 3)
    })


def new_seed():
    return np.random.randint(0, np.iinfo(np.int32).max)


def t(xs):
    return np.swapaxes(xs, -2, -1)


def get_flops(fn, *args, **kwargs):
    """Borrowed from flax.nn.tabulate"""
    e = fn.lower(*args, **kwargs)
    cost = e.cost_analysis()
    if cost is None:
        return 0
    flops = int(cost['flops']) if 'flops' in cost else 0
    return flops


class Finite:
    def __init__(self, task, data_size, seed=None) -> None:
        self.task = task
        self.data_size = data_size
        self.batch_size = self.task.batch_size
        self.task.batch_size = data_size   # dirty trick (we're all adults here)

        self.data = next(self.task)
        del self.task                      # task is consumed

        self.rng = np.random.default_rng(seed)
    
    def __next__(self):
        idxs = self.rng.choice(self.data_size, self.batch_size, replace=True)
        return self.data[0][idxs], self.data[1][idxs]

    def __iter__(self):
        return self


@dataclass
class Case:
    name: str
    config: dataclass
    train_task: Iterable | None = None
    test_task: Iterable | None = None
    train_args: dict = field(default_factory=dict)
    state: list = None
    hist: list = None
    info: dict = field(default_factory=dict)

    def run(self):
        self.state, self.hist = train(self.config, data_iter=self.train_task, test_iter=self.test_task, **self.train_args)
    
    def get_flops(self):
        train_args = self.train_args
        loss = train_args.get('loss', None)
        return get_flops(train_step, self.state, next(self.train_task), loss=loss)
    
    def eval(self, task, key_name='eval_acc'):
        xs, ys = next(task)
        logits = self.state.apply_fn({'params': self.state.params}, xs)

        if len(logits.shape) > 1:
            preds = logits.argmax(-1)
        else:
            preds = (logits > 0).astype(float)

        eval_acc = np.mean(ys == preds)
        self.info[key_name] = eval_acc
    
    def eval_mse(self, task, key_name='eval_mse'):
        xs, ys = next(task)
        ys_pred = self.state.apply_fn({'params': self.state.params}, xs)
        mse = np.mean((ys - ys_pred)**2)

        self.info[key_name] = mse


@ dataclass
class KnnCase:
    name: str
    config: dataclass
    train_task: Iterable | None = None
    info: dict = field(default_factory=dict)

    def run(self):
        data_size = self.train_task.data[0].shape[0]
        self.config.xs = self.train_task.data[0].reshape(data_size, -1)
        self.config.ys = self.train_task.data[1]
        self.model = self.config.to_model()

    def eval(self, task, key_name='eval_acc'):
        xs, ys = next(task)
        probs = self.model(xs)
        eval_acc = np.mean(ys == probs.argmax(-1))
        self.info[key_name] = eval_acc

    def eval_mse(self, task, key_name='eval_mse'):
        xs, ys = next(task)
        ys_pred = self.model(xs)
        mse = np.mean((ys - ys_pred)**2)

        self.info[key_name] = mse


def eval_cases(all_cases, eval_task, key_name='eval_acc', use_mse=False, ignore_err=False):
    try:
        len(eval_task)
    except TypeError:
        eval_task = itertools.repeat(eval_task)

    for c, task in tqdm(zip(all_cases, eval_task), total=len(all_cases)):
        try:
            if use_mse:
                c.eval_mse(task, key_name)
            else:
                c.eval(task, key_name)
        except Exception as e:
            if ignore_err:
                continue
            else:
                raise e


def split_cases(all_cases, run_split):
    run_idx = sys.argv[1]
    try:
        run_idx = int(run_idx) % run_split
    except ValueError:
        print(f'warn: unable to parse index {run_idx}, setting run_idx=0')
        run_idx = 0

    print('RUN IDX', run_idx)
    all_cases = np.array_split(all_cases, run_split)[run_idx]
    return list(all_cases)


def summon_dir(path: str, clear_if_exists=False):
    new_dir = Path(path)
    if not new_dir.exists():
        new_dir.mkdir(parents=True)
    elif clear_if_exists:
        for item in new_dir.iterdir():
            shutil.rmtree(item)
    
    return new_dir


def collate_dfs(df_dir, show_progress=False):
    pkl_path = Path(df_dir)
    dfs = []
    if show_progress:
        for f in tqdm(list(pkl_path.iterdir())):
            dfs.append(pd.read_pickle(f))
    else:
        dfs = [pd.read_pickle(f) for f in pkl_path.iterdir() if f.suffix == '.pkl']
    df = pd.concat(dfs)
    return df


def uninterleave(interl_xs):
    xs = interl_xs[:,0::2]
    ys = interl_xs[:,1::2,[0]]
    xs, x_q = xs[:,:-1], xs[:,[-1]]
    return xs, ys, x_q


def unpack(pack_xs):
    xs = pack_xs[:,:-1,:-1]
    ys = pack_xs[:,:-1,[-1]]
    x_q = pack_xs[:,[-1],:-1]
    return xs, ys, x_q


def estimate_dmmse(task, xs, ys, x_q, sig2=0.05):
    '''
    xs: N x P x D
    ys: N x P x 1
    x_q: N x 1 x D
    ws: F x D
    '''
    ws = task.ws
    
    weights = np.exp(-(1 / (2 * sig2)) * np.sum((ys - xs @ ws.T)**2, axis=1))  # N x F
    probs = weights / (np.sum(weights, axis=1, keepdims=True) + 1e-32)
    w_dmmse = np.expand_dims(probs, axis=-1) * ws  # N x F x D
    w_dmmse = np.sum(w_dmmse, axis=1, keepdims=True)  # N x 1 x D
    return (x_q @ t(w_dmmse)).squeeze()


def estimate_ridge(task, xs, ys, x_q, sig2=0.05):
    n_dims = xs.shape[-1]
    w_ridge = np.linalg.pinv(t(xs) @ xs + n_dims * sig2 * np.identity(n_dims)) @ t(xs) @ ys
    return (x_q @ w_ridge).squeeze()

# NOTE: remove once added to optax upstream
def scale_by_sign():
    def init_fn(params):  # no access to init_empty_state
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        del params
        updates = jax.tree.map(lambda g: jnp.sign(g), updates)
        return updates, state
    
    return optax.GradientTransformation(init_fn, update_fn)


def sign_sgd(learning_rate):
    return optax.chain(
        scale_by_sign(),
        optax.scale_by_learning_rate(learning_rate)
    )