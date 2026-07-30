import numpy as np
from skopt import Optimizer
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import Matern, ConstantKernel as C
from skopt.space import Real

class BayesianEvaluator:
    def __init__(self, x_filtered,y_pure_floats):
        "Initialising optimiyer"

        k01_data= [point[0] for point in x_filtered]
        k02_data= [point[1] for point in x_filtered]
  
        def get_log_padded_bounds(data,pad_margin=0.05):
            valid_data=np.maximum(data,1e-18)
            log_data=np.log10(valid_data)

            log_min, log_max= np.min(log_data), np.max(log_data)
            span=log_max-log_min if log_max > log_min else 1.0

            padded_log_min=log_min-(span*pad_margin)
            padded_log_max=log_max+(span*pad_margin)

            return 10**padded_log_min, 10**padded_log_max

        k01_min, k01_max= get_log_padded_bounds(k01_data)
        k02_min, k02_max= get_log_padded_bounds(k02_data)

        self.search_space = [
            Real(k01_min, k01_max, name='k01'),
            Real(k02_min,k02_max, name='k02'),
        ]

        kernel= C(1.0,(1e-2,1e2))*Matern(
            length_scale=[1.0,1.0],
            length_scale_bounds=[0.1,10.0],
            nu=2.5
        )
        robust_gp= GaussianProcessRegressor(
            kernel=kernel,
            noise=1e-6,
            normalize_y=True,
            n_restarts_optimizer=10, #helping locate local maxima
            random_state=42
        )
        
        self.optimizer= Optimizer(
            dimensions=self.search_space,
            base_estimator=robust_gp,
            acq_func="LCB", #cost min.. or EI
            acq_optimizer="sampling", #preveting stuck in boundaries
            acq_func_kwargs={"kappa":5.0}, #search outside historical cluster
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
        f"Ask skopt for next optimal [factors ]"
        return self.optimizer.ask()
    
    def tell_new_results(self, point, cost_score):
        "Updates GP surface with OGS results"
        clean_point=[float(val) for val in point] #forcing native float
        self.optimizer.tell(clean_point,float(cost_score))
    