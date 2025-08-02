import numpy as np
from src.perturbations.gravity_gradient import GravityGradientPert


class PerturbationManager:
    def __init__(self, config, I):
        self.perturbations = []

        if config.get('gravity_gradient', False):
            self.perturbations.append(GravityGradientPert(I))

    def compute_total_torque(self, **kwargs):
        total = np.zeros(3)
        for p in self.perturbations:
            total += p.compute(**kwargs)
        return total