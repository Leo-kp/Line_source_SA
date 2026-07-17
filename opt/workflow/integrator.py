import os
import shutil
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
import json

import config
import functions as fn
from evaluator import BayesianEvaluator

class OptimizationIntegrator:
    def __init__(self):
        print("[Integrator] Initializing Integration System")

        config.OUT_DIR.mkdir(parents=True, exist_ok=True)
        config.RUN_DIR.mkdir(parents=True, exist_ok=True)
        
        if not config.IS_MESH_DYNAMIC: 
            print("[Integrator] Compiling static baseline meshes in MESH_DIR...")
            config.MESH_DIR.mkdir(parents=True, exist_ok=True)
            self._run_python_sub("mesh.py",[config.ACTIVE_MESH_PATH.as_posix()])

        self.x_history, self.y_history = self._load_morris_history()
        x_clean= [[float(val) for val in point] for point in self.x_history]
        y_clean=[float(val) for val in self.y_history]

        self.evaluator = BayesianEvaluator(x_clean,y_clean)

    def _run_python_sub(self,scritp_name:str,args:list) ->subprocess.CompletedProcess: #subprocess personalised
        cmd=[config.OGS_PYTHON_EXE,scritp_name]+args
        result=subprocess.run(cmd,capture_output=False,text=False)
        return result
    
    def run_ogs_simulation(self, ogs_cmd, debug, log_level,log_filepath):
        """Subprocess for OGS"""

        cmd=list(ogs_cmd)
        if debug:
            cmd+= ["-l",log_level]
            Path(log_filepath).parent.mkdir(parents=True, exist_ok=True)

            with open(log_filepath, "wb") as log_file: #system neutral, raw text-->directly to logtext
                result=subprocess.run(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=False
                )
        else:
            result=subprocess.run(cmd,capture_output=False,text=False)
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
                sf0=float(row['sf0']) #3rd parameter

                cost = fn.objective_function(run_dict,field_data)

                x_history.append([pjack,wr,sf0])
                y_history.append(float(cost))
            except  Exception as e:
                print(f"Warning: skipping corrupted history file {file_path.name}. Error {e}")

        print(f"[integrator] Succesfully matched {len(x_history)} prior Morris samples")
        return x_history,y_history
    
    def run_optimization_loop(self, max_iterations: int=20, cost_tolerance:float=1e-4,target_cost:float=10000):#variable conditioning
        """Optimization loop with triple-conditioning
        :param max_iterations: condition 1--> hard maximum looping limit
        :param cost_tolerance: condition 2--> relative  improvement threshold
        :param target_cost:    condition 3--> absolute target cost"""

        print(f"[Integrator] Starting optimization loop ({max_iterations}iterations)")
        successful_iterations=[]
        cost_history=[]

        field_data= pd.read_csv(config.FIELD_DATA_PATH, header=0)
        factors_payload=config.factors_payload  #***************************************************

        for iteration in range(1,max_iterations+1):
            print(f"\n--- interation {iteration}/{max_iterations} ---")
            suggested_point=self.evaluator.ask_next_point()
            pjack_val, wr_val,sf0_val= suggested_point[0], suggested_point[1],suggested_point[2]
            print(f"[Loop] Testing Parameters pjack: {pjack_val:.4f}, wr: {wr_val:.4f},sf0: {sf0_val:.3e}") ##*****************************

            if config.OUT_DIR.exists():
                shutil.rmtree(config.OUT_DIR)
            config.OUT_DIR.mkdir(parents=True, exist_ok=True)

            if config.IS_MESH_DYNAMIC: 
                print("[Integrator] Compiling static baseline meshes in MESH_DIR...")
                config.MESH_DIR.mkdir(parents=True, exist_ok=True)
                self._run_python_sub("mesh.py",[config.ACTIVE_MESH_PATH.as_posix()])

            
            factors_payload['pjack']=pjack_val
            factors_payload['wr']=wr_val
            factors_payload['sf0']=sf0_val #****************************************************

            calculated_k=fn.calculate_keff(factors_payload)
            factors_payload['keff']=calculated_k.tolist() if hasattr(calculated_k, 'tolist') else calculated_k 
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

            print("[Loop] Executing OpenGeosys simulation...") #pure Python --> crossplatform
            ogs_cmd=[
                 config.OGS_BINARY, #direct ogs.exe-->crossplatform
                 config.RUNTIME_PRJ.as_posix(),
                 "-o", config.OUT_DIR.as_posix()
             ]
            
            log_filepath=(Path(config.OUT_DIR)/config.LOG_FILENAME).as_posix()

            sim_result=self.run_ogs_simulation(
                ogs_cmd=ogs_cmd,
                debug=config.DEBUG,
                log_level=config.LOG_LEVEL,
                log_filepath=log_filepath
            )

            if sim_result.returncode !=0:
                print(f"CRITICAL_ERROR: OGS simulation failed at iteration {iteration}!")
                continue #safeguard ? continuing if breaks

            print("[Loop] Extracting data")
            try:
                live_npy_path=config.RUN_DIR/f"iter_{iteration}_data.npy" 
                sub_res=self._run_python_sub("probing.py",[config.OUT_DIR.as_posix(),live_npy_path.as_posix()])

                if sub_res.returncode !=0:
                    raise RuntimeError(f"probing.py failed: {sub_res.stderr}")
                
                extracted_bundle=np.load(live_npy_path,allow_pickle=True).item()
                cost_score=fn.objective_function(extracted_bundle,field_data)
                extracted_bundle["metadata"]= {'pjack':pjack_val,'wr':wr_val, 'sf0': sf0_val, 'iteration':iteration,'cost':cost_score} #*************
                np.save(live_npy_path, extracted_bundle)
                
                print(f"[Loop] Iteration Result Mismatch Cost: {cost_score:.6f}")

            except Exception as ex:
                print(f"Error during post-processing iteration {iteration}: {ex}")
                continue 
            
            self.evaluator.tell_new_results(suggested_point,cost_score)
            successful_iterations.append(iteration+1)
            cost_history.append(cost_score)
           
            if cost_score<=target_cost:
                print(f"[Loop] Target cost achieved, cost: {cost_score:.4f}")
                print("Finishing looping by condition 3")
                break

            if len(cost_history)>=2:
                relative_change=abs(cost_score-cost_history[-2])/cost_history[-2]
                if relative_change<cost_tolerance:
                    print(f"[Loop] Convergence, relative change: {relative_change:.4f}")
                    print("Finishing looping by condition 2")
                    break
        else: #completing full iterations
            print("[Integrator] Optimization Loop completed by iterations")
            print("Finishing looping by condition 1")

if __name__=="__main__":
    print("Initialization")
    Integrator=OptimizationIntegrator()

    print(f"Loaded config: DEBUG={config.DEBUG}, LOG_LEVEL={config.LOG_LEVEL}")
    print(f"Output directory: {config.OUT_DIR}")

    try:
        print("[Integrator] Starting optimization loop...")
        result=Integrator.run_optimization_loop(max_iterations=20)

    except Exception as error:
        print("\n---pipeline failed---")
        print(f"Error details: {error}")

