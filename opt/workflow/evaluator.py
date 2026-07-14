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

        pjack_min, pjack_max= min(pjack_data), max(pjack_data)
        wr_min, wr_max= min(wr_data), max(wr_data)

        pjack_padding=(pjack_max - pjack_min)*0.01
        wr_padding= (wr_max - wr_min)*0.01

        self.search_space=[ #to avoid point outside search space is bounded
            Real(pjack_min-pjack_padding, pjack_max+pjack_padding,name='pjack'),
            Real(wr_min-wr_padding, wr_max+wr_padding,name='wr')
        ]

        # robust_gp= GaussianProcessRegressor(
        #     kernel=Matern(nu=2.5),
        #     alpha=1e-6,
        #     noise="gaussian",
        #     normalize_y=True,
        #     random_state=42
        # )
        
        # self.optimizer= Optimizer(
        #     dimensions=self.search_space,
        #     base_estimator=robust_gp,
        #     acq_func="EI",
        #     random_state=42,
        # )

        self.optimizer= Optimizer(
            dimensions=self.search_space,
            base_estimator="GP",
            acq_func="EI",
            random_state=42,
        )

        try:
            self.optimizer.tell(x_filtered,y_pure_floats,fit=True)
            print("[Evaluator] Success: Optimizer successfully integrated the data...")
        except Exception as e:
            print(f"Crash error: {e}")
    
    def ask_next_point(self):
        "Ask skopt for next optimal [pjack,wr]"
        return self.optimizer.ask()
    
    def tell_new_results(self, point, cost_score):
        "Updates GP surface with OGS results"
        clean_point=[float(val) for val in point] #forcing native float
        self.optimizer.tell(clean_point,float(cost_score))
    