from hbmep.config import Config
from hbmep.nn import functional as F
from hbmep.model import BoundedOptimization
from hbmep.model.utils import Site as site


class RectifiedLogistic(BoundedOptimization):
    NAME = "rectified_logistic"

    def __init__(self, config: Config):
        super(RectifiedLogistic, self).__init__(config=config)
        self.solver = "Nelder-Mead"
        self.functional = F.rectified_logistic
        self.named_params = [site.a, site.b, site.L, site.ell, site.H]
        self.bounds = [(1e-9, 30.), (1e-9, 10), (1e-9, 10), (1e-9, 10), (1e-9, 10)]
        self.informed_bounds = [(2, 8), (1e-3, 5.), (1e-4, .1), (1e-2, 5), (.5, 5)]
        self.num_points = 1000
        self.num_iters = 100
        self.n_jobs = -1


class Logistic5(BoundedOptimization):
    NAME = "logistic5"

    def __init__(self, config: Config):
        super(Logistic5, self).__init__(config=config)
        self.solver = "Nelder-Mead"
        self.functional = F.logistic5
        self.named_params = [site.a, site.b, site.v, site.L, site.H]
        self.bounds = [(1e-9, 30.), (1e-9, 10), (1e-9, 10), (1e-9, 10), (1e-9, 10)]
        self.informed_bounds = [(2, 8), (1e-3, 5.), (1e-3, 5.), (1e-4, .1), (.5, 5)]
        self.num_points = 1000
        self.num_iters = 100
        self.n_jobs = -1


class Logistic4(BoundedOptimization):
    NAME = "logistic4"

    def __init__(self, config: Config):
        super(Logistic4, self).__init__(config=config)
        self.solver = "Nelder-Mead"
        self.functional = F.logistic4
        self.named_params = [site.a, site.b, site.L, site.H]
        self.bounds = [(1e-9, 30.), (1e-9, 10), (1e-9, 10), (1e-9, 10)]
        self.informed_bounds = [(2, 8), (1e-3, 5.), (1e-4, .1), (.5, 5)]
        self.num_points = 1000
        self.num_iters = 100
        self.n_jobs = -1


class RectifiedLinear(BoundedOptimization):
    NAME = "rectified_linear"

    def __init__(self, config: Config):
        super(RectifiedLinear, self).__init__(config=config)
        self.solver = "Nelder-Mead"
        self.functional = F.rectified_linear
        self.named_params = [site.a, site.b, site.L]
        self.bounds = [(1e-9, 30.), (1e-9, 10), (1e-9, 10)]
        self.informed_bounds = [(2, 8), (1e-3, 5.), (1e-4, .1)]
        self.num_points = 1000
        self.num_iters = 100
        self.n_jobs = -1
