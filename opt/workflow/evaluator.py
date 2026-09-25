import numpy as np
import pandas as pd
from skopt import Optimizer
from skopt.space import Real

import joblib
from sklearn.pipeline import Pipeline

class BayesianEvaluator:
    def __init__(self, x_filtered:pd.DataFrame | pd.Series ,y_filtered : list | list[list] , pipeline_path = None):
        "Initialising optimizer"
        if pipeline_path is not None:
                pipeline = joblib.load(pipeline_path)
        else:
                raise Exception("[Error: Please, load a valid path to pipeline GP]")

        self.pipeline_features = list(pipeline.feature_names_in_)
        x_filtered["log_sf0"] = np.log(x_filtered['sf0'], dtype = float)

        pjack_data= x_filtered['pjack'] # 
        wr_data= x_filtered['wr'] 
        L_data= x_filtered['L']
        log_sf0_data= x_filtered['log_sf0']  #***************************

        def get_padded_bounds(data, lower_limit=None, upper_limit=None):
            d_min, d_max = min(data), max(data)
            padding = 0#(d_max - d_min) * 0.01 if d_max > d_min else 1e-6

            padded_min = d_min - padding
            padded_max = d_max + padding

            if lower_limit is not None:
                padded_min=max(lower_limit, padded_min)
            if upper_limit is not None: 
                padded_max= min(upper_limit,padded_max)

            return padded_min, padded_max

        pjack_min, pjack_max= get_padded_bounds(pjack_data)
        wr_min, wr_max= get_padded_bounds(wr_data)
        L_min, L_max= get_padded_bounds(L_data)
        sf0_min, sf0_max= get_padded_bounds(log_sf0_data,lower_limit=np.log(2.0926206997084548e-10))

        bounds_dict = {
                        "pjack":     Real(pjack_min, pjack_max, name='pjack'),
                        "wr":        Real(wr_min, wr_max, name='wr'),
                        "L":         Real(max(0.4,L_min), L_max, name='L'),
                        "log_sf0":   Real(sf0_min, sf0_max, name='log_sf0')
        }

        self.search_space = [bounds_dict[col] for col in self.pipeline_features ]

        self.pre_trained_gp = pipeline.named_steps['gp']

        x_ = x_filtered[self.pipeline_features]   #variable transformed
        x_ready = [[float(val) for val in col] for col in x_.values]
        y_ready= y_filtered

        self.optimizer= Optimizer(
            dimensions=self.search_space,
            base_estimator=self.pre_trained_gp,
            acq_func="EI",            #"EI", general
            acq_optimizer="sampling", #"sampling", global
            n_initial_points=0,# fixing preload data in history
            random_state=42,
        )

        try:
            
            self.optimizer.tell(x_ready,y_ready,fit=True) #fit=True refitting of GP, false: using GP pretrained combined with append()

            print("[Evaluator] Success: Optimizer successfully integrated the data...")
        except Exception as e:
            import traceback
            print(f"Crash error: {e}")
            print("Exact Crash Cause:")
            traceback.print_exc()

        self.last_raw_point = None
    
    def ask_next_point(self):
        "Ask skopt for next optimal "
        self.last_raw_point = self.optimizer.ask()
        factors_dict = dict(zip(self.pipeline_features, self.last_raw_point))
        physical_dict = factors_dict.copy()

        physical_dict ={ 
            ("sf0" if k == "log_sf0" else k) : (float(np.exp(v)) if k == "log_sf0" else v) for k, v in factors_dict.items()
        } #order replacement and transform

        return physical_dict
    
    def tell_new_results(self, cost_score, point=None):
        "Updates GP surface with OGS results"

        if point is not None:
            raw_point = self.convert_physical_to_raw(point)
        else:
            raw_point = self.last_raw_point

        clean_raw_point=[float(val) for val in raw_point] #forcing native float
        clean_cost = float(cost_score)

        self.optimizer.tell(clean_raw_point,clean_cost,fit=True)


    def convert_physical_to_raw(self, physical_dict: dict):
        d = physical_dict.copy()
        if "sf0" in d:
            d["log_sf0"] = float(np.log(d.pop("sf0"))) #order replacement and transform

        return [float(d[col]) for col in self.pipeline_features]

    def current_gp(self):
         """return lates GP"""
         if len(self.optimizer.models) > 0:
              return self.optimizer.models[-1]
         return self.pre_trained_gp

    def current_pipeline(self):
         if self.scaler is None:
              raise ValueError("No scaler found")

         return Pipeline([
              ("scaler", self.scaler),
              ("gp", self.current_gp())
         ])
         
##------------------------------------------------------------------------
# if __name__=="__main__":
#     import pandas as pd

#     print("runing test")

#     raw_df = pd.DataFrame(
#         {"pjack": [10, 20, 30, 40, 50],
#         "wr": [0.5, 0.6, 0.7, 0.8, 0.9],
#         "L": [1, 2, 3, 4, 6],
#         "sf0": [1e-2, 2e-2, 3e-3, 4e-3, 5e-3]
#     })

#     raw_df['log_sf0'] = np.log(raw_df['sf0'])
#     working_df = raw_df[["pjack", "wr", "L", "log_sf0"]]

#     raw_y = pd.Series([150, 106.0, 200, 350.0, 363.3])

 
#     x_load_h= raw_df #working_df.values.astype(float).tolist()#[[float(val) for val in raw_df[col]] for col in raw_df]
#     y_load_h= raw_y.astype(float).to_list()#[float(val) for val in raw_y]

#     evaluator = BayesianEvaluator(
#         x_filtered= x_load_h,
#         y_filtered= y_load_h,
#         pipeline_path= r"C:\Users\penack5q\Activities\OpenGeoSys\github\Benchmarks\Line_source_SA\pipelines_others\gp_tunning_opt5_v15_pipeline.pkl"
#     )

#     suggested = evaluator.ask_next_point()
#     cost_score = 142.5

#     evaluator.tell_new_results(cost_score, suggested)
