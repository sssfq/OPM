from TrainEvalFun import trainevalfun
from bayes_opt import BayesianOptimization
from bayes_opt import SequentialDomainReductionTransformer
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as kernels

pbounds = {
    'batch_size':(5,40),
    'learning_rate':(1e-4,1e-2),
    'num_epochs':(3,10)
}

bounds_transformer = SequentialDomainReductionTransformer()

bayesianoptimizer = BayesianOptimization(
    f = trainevalfun,
    pbounds = pbounds,
    random_state =1,
    bounds_transformer=bounds_transformer
)
logger = JSONLogger(path="./logs.json")
bayesianoptimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
bayesianoptimizer.probe(params=[20,1e-3,5],lazy=True)
bayesianoptimizer.maximize(init_points=4,n_iter=10,kernel=kernels.Matern(nu=2.5))

