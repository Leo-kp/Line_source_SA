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
    def __init__(self):
        print("[Integrator] Initializing Integration System")

        config.RUN_DIR.mkdir(parents=True, exist_ok=True)
        config.MESH_DIR.mkdir(parents=True, exist_ok=True)
        mesh.generate_optimization_mesh() #review position (here as static)
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

        history_files=list(config.MORRIS_RAW_DATA_DIR.glob("*.npy"))

        if not history_files:
            raise FileNotFoundError(f"No historical morris found in {config.MORRIS_RAW_DATA_DIR}. Cannot warm-start")
        
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
        print(f"[Integrator] Starting optimization loop ({max_iterations}iterations)")


        field_data= np.load(config.FIELD_DATA_PATH, allow_pickle=True).item()

        for iteration in range(1,max_iterations+1):
            print(f"\n--- interation {iteration}/{max_iterations}")
            suggested_point=self.evaluator.ask_next_point()
            pjack_val, wr_val= suggested_point[0], suggested_point[1]
            print(f"[Loop] Testing Parameters pjack: {pjack_val:.4f}, wr: {wr_val:.4f}")

            #mesh.generate_optimization_mesh() #review position (here as dynamic)

            factors_payload={ #initiating payload manually
                    'k01':2e-15,
                    'k02':1.0,
                    'sma':2.0926206997084548e-10,
                    'L':0.5,
                    'pjack':pjack_val,
                    'wr':wr_val,
                    'b_dim':1.0,
                    'sf0':2.8e-4,
                    'p1':1.0,
                    'p2':5.0e6,
                    'keff':1.0
            }

            factors_payload=fn.calculate_keff(factors_payload) 
            prj_mod.temp_prj(config.TEMPLATE_PRJ, config.RUNTIME_PRJ, factors_payload) 

            if config.OUT_DIR.exists():
                shutil.rmtree(config.OUT_DIR)
            config.OUT_DIR.mkdir(parents=True, exist_ok=True)

            print("[Loop] Executing OpenGeosys simulation...")
            ogs_cmd=[
                 config.OGS_BIN_DIR, #review path and execution ------------------
                 config.RUNTIME_PRJ.as_posix(),
                 "-o", config.OUT_DIR.as_posix()
             ]
            
            sim_result= subprocess.run(ogs_cmd,capture_ouput=True,text=True)
            if sim_result.returncode !=0:
                print(f"CRITICAL_ERROR: OGS simulation failed at iteration {iteration}!")
                print(sim_result.stderr)
                continue #safeguard continuing if breaks

            print("[Loop] Extracting data")
            try:
                extracted_bundle=probing.extract_values(config.OUT_DIR)
                extracted_bundle["metadata"]= {'pjack':pjack_val,'wr':wr_val, 'iteration':iteration}

                live_npy_path=config.RUN_DIR/f"iter_{iteration}_data.pny" #review output npy---------------------------------------------------------------------
                np.save(live_npy_path, extracted_bundle)

                cost_score=fn.objective_function(extracted_bundle,field_data)
                print(f"[Loop] Iteration Result Mismatch Cost: {cost_score:.6f}")

                self.evaluator.tell_new_results(suggested_point,cost_score)

            except Exception as ex:
                print(f"Error during post-processing iteration {iteration}: {ex}")
                continue 

            print("\n[Integrator] Optimization routine completed sucessfully.")

if __name__=="__main__":
    runner=OptimizationIntegrator()
    runner.run_optimization_loop(max_iterations=15)




