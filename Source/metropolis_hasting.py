import importlib
from pathlib import Path
from typing import Any, Dict
import numpy as np
import yaml


def load_config(config: Path) -> Dict[str, Any]:
    path = Path(config)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"No YAML configuration file found at {path}")


class MetroHaste:
    def __init__(self, config: Path, model_function):
        """
        Constructor of Metropolis Hasting Algorithm for given function
        :param config: Configuration file containing details of run
        :param model_function: Gravitational Wave function
        """
        self.cfg = load_config(config)
        self.rng = np.random.default_rng(self.cfg.get("seed"))

        fun_cfg = self.cfg["functions"]
        mod = importlib.import_module(fun_cfg["module"])
        self.model = model_function
        self.loglike = getattr(mod, fun_cfg.get("loglike_fn", "log_likelihood"))

        self.param_names = list(self.cfg["param_names"])
        self.dim = len(self.param_names)

        self.bounds = self.cfg["bounds"]
        self.scales = np.asarray(self.cfg["proposal_scales"], dtype=float)
        self.n_samples = int(self.cfg.get("n_samples"))
        self.burn_in = int(self.cfg.get("burn_in"))
        self.thin = int(self.cfg.get("thin"))

        init = self.cfg.get("init")
        self.theta0 = np.array([init[n] for n in self.param_names], dtype=float)

    def MH_Solver(self, datapoints):
        theta = self.theta0.copy()
        # Use observed y in column 1
        f_data = datapoints[:, 1]
        f_prior_prev = self.model(*self.theta0)
        logL = self.loglike(f_data, f_prior_prev)
        chain = []
        accepted = 0
        for i in range(self.n_samples):
            step = self.rng.normal(0.0, self.scales, size=self.dim)
            theta_next = theta + step
            if not self._in_support(theta_next):
                if i >= self.burn_in and ((i - self.burn_in) % self.thin == 0):
                    chain.append(theta.copy())
                continue
            f_prior_next = self.model(*theta_next)
            logL_new = self.loglike(f_data, f_prior_next)
            A = np.exp(min(0.0, float(logL_new - logL)))
            if self.rng.random() < A:
                theta = theta_next
                logL = logL_new
                accepted += 1
            if i >= self.burn_in and ((i - self.burn_in) % self.thin == 0):
                chain.append(theta.copy())
        chain = np.array(chain)
        median = np.median(chain, axis=0)
        diag = {
            "acceptance_rate": accepted / self.n_samples,
            "predicted_parameters": median
        }
        return chain, diag

    def _in_support(self, theta: np.ndarray) -> bool:
        for v, n in zip(theta, self.param_names):
            b = self.bounds[n]
            if not (float(b["min"]) < float(v) < float(b["max"])):
                return False
        return True
