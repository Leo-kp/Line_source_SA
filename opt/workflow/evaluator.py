import os
import subprocess
import numpy as np
from contextlib import chdir

from prj_mod import calculate_keff, temp_prj
from interp import objective_function


def run_iteration(factors, datafield, RUN_DIR, prj_name, OGS_BIN_DIR):
