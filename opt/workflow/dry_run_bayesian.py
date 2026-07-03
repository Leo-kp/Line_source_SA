import numpy as np
from  skopt import gp_minimize
from skopt.space import Real
import pandas as pd

from pathlib import Path


cwd=Path.cwd()

def run_dry_opt(cwd=cwd):

    try:
        version = "v12" 
        x_ =pd.read_csv(cwd.parent.parent/f"results_{version}"/f"morris_samples_{version}.csv")
        
        raw_paths= list((cwd.parent.parent/f"results_{version}"/"results/raw_data").glob("*.npy"))
        file_paths=sorted(
            raw_paths,
            key=lambda p: int(p.stem.split('_')[2])
        )
        y_list= [np.load(file,allow_pickle=True).item() for file in file_paths]

    except FileNotFoundError as e:
        print(f"No hystorical data {e}")
    
    print("Data Loaded")

if __name__=="__main__":
    run_dry_opt(cwd)