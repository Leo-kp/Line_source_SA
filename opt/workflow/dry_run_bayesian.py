import numpy as np
from  skopt import gp_minimize
from skopt import Optimizer
from skopt.space import Real
import pandas as pd

from pathlib import Path
from functions import objective_function
import random

cwd=Path.cwd()
fieldata= pd.read_csv(cwd/'BH10_20180718_40.6_SR_v2.csv',header=0)

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

    y_init=[]
    y_array=np.array(y_list)
    for idx, run_dict in enumerate(y_array):
        cost= objective_function(run_dict,fieldata)
        y_init.append(cost)

    print(f"processed y_init shape: ({len(y_init)},) - Ready for optimizer.tell(x_,y_init)")
    print(f"Total processed e: {len(y_init)}")
    print(f"Data types in y_init: {set(type(val) for val in y_init)}")

    if hasattr(x_,"columns"):
        x_filtered= x_[["pjack","wr"]].values.tolist()
        print(f"Extracted 'pjack' and 'wr' from df. Shape: ({len(x_filtered)},2) ")
    else: 
        print("X are not dataframe with column names")

    # pjack_data= [point[0] for point in x_filtered]
    # wr_data= [point[1] for point in x_filtered]

    # pjack_min, pjack_max= min(pjack_data), max(pjack_data)
    # wr_min, wr_max= min(wr_data), max(wr_data)

    # pjack_padding=(pjack_max - pjack_min)*0.01
    # wr_padding= (wr_max - wr_min)*0.01

    # search_space=[ #to avoid point outside search space
    #     Real(pjack_min-pjack_padding, pjack_max+pjack_padding,name='pjack'),
    #     Real(wr_min-wr_padding, wr_max+wr_padding,name='wr')
    # ]

    # optimizer= Optimizer(
    #     dimensions=search_space,
    #     base_estimator="GP",
    #     acq_func="EI",
    #     random_state=42
    # )

    # print(f"Training surrogate model 2D space (pjack,wr)...")
    # optimizer.tell(x_filtered,y_init,fit=True)
    # print("optimizer ready")

    # next_point=optimizer.ask()
    # print(f"\nrecomended experiment")
    # print(f"suggested pjack: {next_point[0]:.4f}")
    # print(f"suggested wr: {next_point[1]:.4f}")

    #1
    # min_idx =int(np.argmin(y_init))
    # max_idx =int(np.argmax(y_init))

    # print(f"\nBest run check")
    # print(f"run index: {min_idx}")
    # print(f"input par: {x_.iloc[min_idx]}")
    # print(f"Mismatch y par: {y_init[min_idx]:.4f}")

    # print(f"\nWorst run check")
    # print(f"run index: {max_idx}")
    # print(f"input par: {x_.iloc[max_idx]}")
    # print(f"Mismatch y par: {y_init[max_idx]:.4f}")

    # print("random verification")
    # random_idx=random.randint(0,len(y_init)-1)
    # print(f"selected random idx verification: {random_idx}")
    # raw_sample_dir= y_list[random_idx]

    # manual_cost=objective_function(raw_sample_dir,fieldata)
    # loop_cost=y_init[random_idx]

    # print(f"loop-calculated mistmatch: {loop_cost}")
    # print(f"Manual calculated Mistmatch: {manual_cost}")

    #0
    # print("verify x")
    # print(f"Type: {type(x_)}")
    # print(f"Dimensions: {x_.shape}")
    # print(f"Columns: {list(x_.columns)}")
    # print(f"Missing/NaN: {x_.isnull().sum().sum()}")
    # print(x_.head(3))

    # print("verify y")
    # print(f"Original Type: {type(y_list)}")
    # print(f"List Length: {len(y_list)}")
    # y_array=np.array(y_list)

    # print(f"converted shape: {y_array.shape}")
    # print(f"data type: {y_array.dtype}")

    # if y_array.ndim>1:
    #     print(f"target dim: {y_array.ndim}")
    #     print(f"Shape of element 0: {y_array[0].shape}")
    # else:
    #     print(f"target dim: 1D flat vector, perfect for optimizer")

    # print(y_array[0])
if __name__=="__main__":
    run_dry_opt(cwd)


