"""
Common utilities used across all experiments

author: William Tong (wtong@g.harvard.edu)
"""

from dataclasses import dataclass, field
import itertools
from typing import Callable, Iterable

import numpy as np
from tqdm import tqdm

import sys
sys.path.append('../')
from train import train

def new_seed():
    return np.random.randint(0, np.iinfo(np.int32).max)

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



def eval_cases(all_cases, eval_task, key_name='eval_acc', ignore_err=False):
    try:
        len(eval_task)
    except TypeError:
        eval_task = itertools.repeat(eval_task)

    for c, task in tqdm(zip(all_cases, eval_task), total=len(all_cases)):
        try:
            c.eval(task, key_name)
        except Exception as e:
            if ignore_err:
                continue
            else:
                raise e