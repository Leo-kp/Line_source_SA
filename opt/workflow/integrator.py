import os
import shutil
import subprocess
import numpy as np
from pathlib import Path

import config
import functions as fn
import prj_mod
import mesh
import probing 
from evaluator import BayesianEvaluator

class OptimizationIntegrator:
    def __init_(self):
        print("[Integrator] Initializing Integration System")

        config.RUN_DIR.mkdir(parents=True, exist_ok=True)
        config.MESH_DIR.mkdir(parents=True, exist_ok=True)

        x_history, y_history = self._load_morris_history()

        self.evaluator = BayesianEvaluator()

    def _load_morris_history(self):
        """
        Gather exsiting Morris analysis (npy files)
        computes mismatch cost for each run against field data
        """
        print("[Integrator] Parsing Morris screening history")
        x_history=[]
        y_history=[]

        history_files=list(config.HISTORZ_DIR.glob("*.npy"))

        if not history_files:
            raise FileNotFoundError(f"No historical morris found in {config.HISTORY_DIR}. Cannot warm-start")
        
        field_data= np.load(config.FIELD_DATA_PATH, allow_pickle=True).item()

        for file_path in history_files:
            try:
                run_dict= np.load(file_path, allow_pickle=True).item()
                pjack= run_dict['metadata']['pjack']
                wr=run_dict['metadata']['wr']

                cost = fn.objective_function(run_dict,field_data)

                x_history.append([pjack,wr])
                y_history.append(float(cost))
            except  Exception as e:
                print(f"Warning: skipping corrupted history file {file_path.name}. Error {e}")

        print(f"[integrator] Succesfully matched {len(x_history)} prior Morris samples")
    
    def run_optimization_loop(self, max_iterations: int=20):
        """Executes live Bayesian operations"""
        print(f"\n--- interation {iteration}/{max_iterations}")

        field_data= np.load(config.FIELD_DATA_PATH, allow_pickle=True).item()

        for iteration in range(1,max_iterations+1):
            