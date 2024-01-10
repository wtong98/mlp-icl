"""
Common utilities used across all experiments

author: William Tong (wtong@g.harvard.edu)
"""

from dataclasses import dataclass, field
import itertools
from pathlib import Path
import shutil
from typing import Callable, Iterable

import numpy as np
from tqdm import tqdm

import sys
sys.path.append('../')
from train import train

def new_seed():
    return np.random.randint(0, np.iinfo(np.int32).max)


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
    
    def eval(self, task, key_name='eval_acc'):
        xs, ys = next(task)
        logits = self.state.apply_fn({'params': self.state.params}, xs)
        preds = logits.argmax(axis=1)
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
    task_class: Callable
    data_size: int
    seed: int
    task_args: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)

    def run(self):
        self.task = self.task_class(data_size=self.data_size, seed=self.seed, **self.task_args)
        self.config.xs = self.task.data[0].reshape(self.data_size, -1)
        self.config.ys = self.task.data[1]
        self.model = self.config.to_model()

    def eval(self, task, key_name='eval_acc'):
        xs, ys = next(task)
        probs = self.model(xs)
        eval_acc = np.mean(ys == probs.argmax(-1))
        self.info[key_name] = eval_acc


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


def summon_dir(path: str, clear_if_exists=False):
    new_dir = Path(path)
    if not new_dir.exists():
        new_dir.mkdir(parents=True)
    elif clear_if_exists:
        for item in new_dir.iterdir():
            shutil.rmtree(item)
    
    return new_dir
