import numpy as np
from skopt import Optimizer
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import Matern
from skopt.space import Real

class BayesianEvaluator:
    def __init__(self, x_filtered,y_pure_floats):
        "Initialising optimiyer"

        pjack_data= [point[0] for point in x_filtered]
        wr_data= [point[1] for point in x_filtered]
        L_data= [point[2] for point in x_filtered]
        sf0_data= [point[3] for point in x_filtered] #***************************

        def get_padded_bounds(data, lower_limit=None, upper_limit=None):
            d_min, d_max = min(data), max(data)
            padding = (d_max - d_min) * 0.01 if d_max > d_min else 1e-6

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
        sf0_min, sf0_max= get_padded_bounds(sf0_data,lower_limit=2.0926206997084548e-10)

        self.search_space = [
            Real(pjack_min, pjack_max, name='pjack'),
            Real(wr_min, wr_max, name='wr'),
            Real(max(0.4,L_min), L_max, name='L'),
            Real(sf0_min, sf0_max, name='sf0')
        ]


        robust_gp= GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-4,
            noise="gaussian",
            normalize_y=True,
            random_state=42
        )
        
        self.optimizer= Optimizer(
            dimensions=self.search_space,
            base_estimator=robust_gp,
            acq_func="EI",
            acq_optimizer="sampling", #preveting stuck in boundaries
            n_initial_points=0,# fixing preload data in history
            random_state=42,
        )

        # self.optimizer= Optimizer(
        #     dimensions=self.search_space,
        #     base_estimator="GP",
        #     acq_func="EI",
        #     random_state=42,
        # )

        try:
            self.optimizer.tell(x_filtered,y_pure_floats,fit=True)
            print("[Evaluator] Success: Optimizer successfully integrated the data...")
        except Exception as e:
            import traceback
            print(f"Crash error: {e}")
            print("Exact Crash Cause:")
            traceback.print_exc()
    
    def ask_next_point(self):
        "Ask skopt for next optimal [pjack,wr]"
        return self.optimizer.ask()
    
    def tell_new_results(self, point, cost_score):
        "Updates GP surface with OGS results"
        clean_point=[float(val) for val in point] #forcing native float
        self.optimizer.tell(clean_point,float(cost_score))
    