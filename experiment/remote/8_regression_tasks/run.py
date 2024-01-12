# <codecell>
from flax.serialization import to_state_dict
import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append('../../')
sys.path.append('../../../')
from common import *
from model.knn import KnnConfig
from model.mlp import MlpConfig
from model.poly import PolyConfig
from model.transformer import TransformerConfig
from task.regression import LinearRegression 
    

def run_length_and_spread_experiment():
    n_iters = 5
    train_iters=150_000

    all_cases = []
    for _ in range(n_iters):
        common_task_args = {'seed': new_seed()}
        common_train_args = {'train_iters': train_iters, 'test_iters': 1, 'test_every': 1000, 'loss': 'mse'}

        curr_tasks = [
            Case('MLP', MlpConfig(n_out=1, n_layers=3, n_hidden=256), train_args=common_train_args),
            Case('Transformer', TransformerConfig(n_out=1, n_layers=3, n_heads=4, n_hidden=256), train_args=common_train_args),
            Case('MNN', PolyConfig(n_out=1, n_layers=1, n_hidden=256), train_args=common_train_args),
        ]

        for case in curr_tasks:
            case.train_task = LinearRegression(**common_task_args)
            case.test_task = LinearRegression(batch_size=1024, **common_task_args)

        all_cases.extend(curr_tasks)


    for case in tqdm(all_cases):
        print('RUNNING', case.name)
        case.run()

    # EVALUATION
    # NOTE: test loss is roughly twice the expected from log output
    x_lims = [5, 10, 20, 40, 80, 160]
    for x_lim in x_lims:
        task = LinearRegression(batch_size=1024, x_range=(-x_lim, x_lim))
        eval_cases(all_cases, task, key_name=f'x_lim_{x_lim}', use_mse=True)

    w_scs = [1, 2, 4, 8, 16, 32]
    for w_scale in w_scs:
        task = LinearRegression(batch_size=1024, w_scale=w_scale)
        eval_cases(all_cases, task, key_name=f'w_scale_{w_scale}', use_mse=True)

    for case in all_cases:
        case.state = to_state_dict(case.state)

    df = pd.DataFrame(all_cases)
    df.to_pickle('res.pkl')

    print('done!')


def run_dimension_experiment():
    n_iters = 5
    train_iters=150_000
    dims = [2, 4, 8, 16, 32, 64, 128]

    all_cases = []
    for _ in range(n_iters):
        for d in dims:
            common_task_args = {'seed': new_seed()}
            common_train_args = {'train_iters': train_iters, 'test_iters': 1, 'test_every': 1000, 'loss': 'mse'}

            curr_tasks = [
                Case('MLP', MlpConfig(n_out=1, n_layers=3, n_hidden=256), train_args=common_train_args),
                Case('Transformer', TransformerConfig(n_out=1, n_layers=3, n_heads=4, n_hidden=256), train_args=common_train_args),
                Case('MNN', PolyConfig(n_out=1, n_layers=1, n_hidden=256), train_args=common_train_args),
            ]

            for case in curr_tasks:
                case.train_task = LinearRegression(n_dims=d, **common_task_args)
                case.test_task  = LinearRegression(n_dims=d, batch_size=1024, **common_task_args)

            all_cases.extend(curr_tasks)


    for case in tqdm(all_cases):
        print('RUNNING', case.name)
        case.run()

    # EVALUATION
    eval_tasks = [c.test_task for c in all_cases]
    eval_cases(all_cases, eval_tasks, use_mse=True)

    for case in all_cases:
        case.state = to_state_dict(case.state)

    df = pd.DataFrame(all_cases)
    df.to_pickle('res_dim.pkl')

    print('done!')
    return df

run_dimension_experiment()