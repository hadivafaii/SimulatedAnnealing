import numpy as np
from typing import List, Tuple


class Config:
    def __init__(
        self,
            domain: dict = None,
            max_steps: int = 1000,
            initial_temperature: float = 1,
            residual_temperature: float = 1e-3,
            w: float = 0.0001,
            sigma: float = 0.8,
            minima_coordinates: List[Tuple] = None,
            minima_depths: List[int] = None,
    ):
        super(Config).__init__()

        if domain is None:
            n = 100
            self.domain = {'x': np.linspace(-5, 5, n), 'y': np.linspace(-5, 5, n)}
        else:
            self.domain = domain

        self.max_steps = max_steps
        self.initial_temperature = initial_temperature
        self.residual_temperature = residual_temperature

        self.w = w
        self.sigma = sigma
        if minima_coordinates is None:
            self.minima_coordinates = [(-1, -2), (-2, 2), (2, 1)]
        else:
            self.minima_coordinates = minima_coordinates
        if minima_depths is None:
            self.minima_depths = [2, 4, 3]
        else:
            self.minima_depths = minima_depths

        assert len(self.minima_coordinates) == len(self.minima_depths)
