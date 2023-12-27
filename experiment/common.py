"""
Common utilities used across all experiments

author: William Tong (wtong@g.harvard.edu)
"""

from dataclasses import dataclass, field
import itertools
from typing import Callable

import numpy as np
from tqdm import tqdm

import sys
sys.path.append('../')
from train import train

def new_seed():
    return np.random.randint(0, np.iinfo(np.int32).max)


def experiment(config, task_class, train_iters=50_000, loss='ce', lr=1e-4, l1_weight=1e-4, **task_kwargs):
    task = task_class(**task_kwargs)
    state, hist = train(config, data_iter=iter(task), loss=loss, test_every=1000, train_iters=train_iters, lr=lr, l1_weight=l1_weight)
    return state, hist


@dataclass
class Case:
    name: str
    config: dataclass
    experiment: Callable | None = None
    experiment_args: dict = field(default_factory=dict)
    state = None
    hist = None
    info: dict = field(default_factory=dict)

    def run(self):
        self.state, self.hist = self.experiment(self.config, **self.experiment_args)
    
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