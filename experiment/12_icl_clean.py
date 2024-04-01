"""
Assembling the final ICL figures for NeurIPS 2024 cleanly
"""

# <codecell>
import jax.numpy as jnp
from flax.serialization import to_state_dict
import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append('../../../')
sys.path.append('../../../../')
from common import *
from model.mlp import MlpConfig, RfConfig, SpatialMlpConfig
from model.transformer import TransformerConfig
from task.regression import FiniteLinearRegression 

# <codecell>
### REGRESSION: smooth interpolation from IWL to ICL
run_id = new_seed()
print('RUN ID', run_id)
