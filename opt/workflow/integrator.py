import os
import shutil
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
import json

import config
import functions as fn
# import prj_mod
# import mesh
# import probing 
from evaluator import BayesianEvaluator

class OptimizationIntegrator:
    def __init__(self):
        print("[Integrator] Initializing Integration System")

        config.OUT_DIR.mkdir(parents=True, exist_ok=True)
        config.RUN_DIR.mkdir(parents=True, exist_ok=True)
        
        if not config.IS_MESH_DYNAMIC: #position (here as static)
            print("[Integrator] Compiling static baseline meshes in MESH_DIR...")
            config.MESH_DIR.mkdir(parents=True, exist_ok=True)
            self._run_python_sub("mesh.py",[config.ACTIVE_MESH_PATH.as_posix()])
            # mesh.generate_optimization_mesh(config.ACTIVE_MESH_PATH) 

        self.x_history, self.y_history = self._load_morris_history()

        self.evaluator = BayesianEvaluator(self.x_history,self.y_history)

    def _run_python_sub(self,scritp_name:str,args:list) ->subprocess.CompletedProcess: #subprocess personalised
        cmd=[config.OGS_PYTHON_EXE,scritp_name]+args
        result=subprocess.run(cmd,capture_output=True,text=True)
        return result

    def _load_morris_history(self):
        """
        Gather exsiting Morris analysis (npy files)
        computes mismatch cost for each run against field data
        """
        print("[Integrator] Parsing Morris screening history")
        x_history=[]
        y_history=[]

        history_files=list(config.MORRIS_RAW_DATA_DIR.glob("*.npy"))
        file_paths=sorted(
            history_files,
            key=lambda p: int(p.stem.split('_')[2])
        )
        if not history_files:
            raise FileNotFoundError(f"No historical morris found in {config.MORRIS_RAW_DATA_DIR}. Cannot warm-start")
        
        try:
            field_data= pd.read_csv(config.FIELD_DATA_PATH, header=0)
            metadata_df= pd.read_csv(config.MORRIS_SAMPLES_CSV, header=0)
        except Exception as e:
            print(f"[CRITICAL ERROR] Failed reading CSV metadata archives: {e}")
            raise e
        
        for idx, file_path in enumerate(file_paths):
            try:
                run_dict= np.load(file_path, allow_pickle=True).item()

                row=metadata_df.iloc[idx]
                pjack= float(row['pjack'])
                wr=float(row['wr'])

                cost = fn.objective_function(run_dict,field_data)

                x_history.append([pjack,wr])
                y_history.append(float(cost))
            except  Exception as e:
                print(f"Warning: skipping corrupted history file {file_path.name}. Error {e}")

        print(f"[integrator] Succesfully matched {len(x_history)} prior Morris samples")
        return x_history,y_history
    
    def run_optimization_loop(self, max_iterations: int=20):
        """Executes live Bayesian operations"""
        print(f"[Integrator] Starting optimization loop ({max_iterations}iterations)")


        field_data= pd.read_csv(config.FIELD_DATA_PATH, header=0)

        for iteration in range(1,max_iterations+1):
            print(f"\n--- interation {iteration}/{max_iterations}")
            suggested_point=self.evaluator.ask_next_point()
            pjack_val, wr_val= suggested_point[0], suggested_point[1]
            print(f"[Loop] Testing Parameters pjack: {pjack_val:.4f}, wr: {wr_val:.4f}")

            if config.OUT_DIR.exists():
                shutil.rmtree(config.OUT_DIR)
            config.OUT_DIR.mkdir(parents=True, exist_ok=True)

            if config.IS_MESH_DYNAMIC: #position (here as static)
                print("[Integrator] Compiling static baseline meshes in MESH_DIR...")
                config.MESH_DIR.mkdir(parents=True, exist_ok=True)
                self._run_python_sub("mesh.py",[config.ACTIVE_MESH_PATH.as_posix()])
                #mesh.generate_optimization_mesh(config.ACTIVE_MESH_PATH) 

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

            calculated_k=fn.calculate_keff(factors_payload)
            factors_payload['keff']=[calculated_k] 

            payload_json=json.dumps(factors_payload)
            prj_res=self._run_python_sub("prj_mod.py",[
                config.TEMPLATE_PRJ.as_posix(),
                config.RUNTIME_PRJ.as_posix(),
                payload_json,
                str(config.IS_MESH_DYNAMIC),
                config.STATIC_MESH_PREFIX
            ])
            if prj_res.returncode!=0:
                print(f"Error in prj_mod: {prj_res.stderr}")
                continue

            # prj_mod.temp_prj(
            #     prj_in=config.TEMPLATE_PRJ,
            #     prj_out= config.RUNTIME_PRJ,
            #     factors= factors_payload,
            #     is_dynamic=config.IS_MESH_DYNAMIC,
            #     static_prefix=config.STATIC_MESH_PREFIX
            # ) 

            print("[Loop] Executing OpenGeosys simulation...") #pure Python --> crossplatform
            ogs_cmd=[
                 config.OGS_BINARY, #direct ogs.exe-->crossplatform
                 config.RUNTIME_PRJ.as_posix(),
                 "-o", config.OUT_DIR.as_posix()
             ]
            
            sim_result= subprocess.run(ogs_cmd,capture_output=True,text=True)
            if sim_result.returncode !=0:
                print(f"CRITICAL_ERROR: OGS simulation failed at iteration {iteration}!")
                print(sim_result.stderr)
                continue #safeguard continuing if breaks

            print("[Loop] Extracting data")
            try:
                live_npy_path=config.RUN_DIR/f"iter_{iteration}_data.npy" #changing order to run subproccess
                sub_res=self._run_python_sub("probing.py",[config.OUT_DIR.as_posix(),live_npy_path.as_posix()])

                if sub_res.returncode !=0:
                    raise RuntimeError(f"probing.py failed: {sub_res.stderr}")
                
                extracted_bundle=np.load(live_npy_path,allow_pickle=True).item()
                # extracted_bundle=probing.extract_values(config.OUT_DIR)
                extracted_bundle["metadata"]= {'pjack':pjack_val,'wr':wr_val, 'iteration':iteration}

                # live_npy_path=config.RUN_DIR/f"iter_{iteration}_data.npy" 
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
    runner.run_optimization_loop(max_iterations=1)




