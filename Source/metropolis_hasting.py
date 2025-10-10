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

    def MH_Solver(self):
        theta = self.theta0.copy()
        logL = self.loglike(theta)

        chain = []
        accepted = 0

        for i in range(self.n_samples):
            step = self.rng.normal(0.0, self.scales, size=self.dim)
            theta_new = theta + step

            if not self._in_support(theta_new):
                if i >= self.burn_in and ((i - self.burn_in) % self.thin == 0):
                    chain.append(theta.copy())
                continue

            logL_new = self.loglike(theta_new)
            A = np.exp(min(0.0, float(logL_new - logL)))

            if self.rng.random() < A:
                theta = theta_new
                logL = logL_new
                accepted += 1

            if i >= self.burn_in and ((i - self.burn_in) % self.thin == 0):
                chain.append(theta.copy())

        chain = np.array(chain)
        diag = {
            "acceptance_rate": accepted / self.n_samples,
            "accepted": accepted,
            "kept": chain.shape[0],
        }
        return chain, diag

    def _in_support(self, theta: np.ndarray) -> bool:
        for v, n in zip(theta, self.param_names):
            b = self.bounds[n]
            if not (float(b["min"]) < float(v) < float(b["max"])):
                return False
        return True
