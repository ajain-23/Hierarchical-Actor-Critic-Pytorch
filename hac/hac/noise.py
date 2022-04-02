from abc import ABC, abstractmethod

import numpy as np


class ActionNoise(ABC):
    """
    Base class for noise applied to DDPG action outputs.
    """

    def __init__(self):
        super(ActionNoise, self).__init__()

    def reset(self) -> None:
        """
        Resetting noise at the end of a learning episode.
        """
        pass

    @abstractmethod
    def __call__(self) -> np.ndarray:
        raise NotImplementedError()


class NormalActionNoise(ActionNoise):
    """
    Action noise from a Gaussian.
    :param mean: the mean value of the noise
    :param std: the scale of the noise (std here)
    """

    def __init__(self, mean, sigma):
        super().__init__()
        self._mu = mean
        self._sigma = sigma

    def __call__(self) -> np.ndarray:
        return np.random.normal(self._mu, self._sigma)

    def __repr__(self) -> str:
        return 'NormalActionNoise(mu={}, sigma={})'.format(self._mu, self._sigma)


class OrnsteinUhlenbeckActionNoise(ActionNoise):
    """
    Action noise from an Ornstein-Uhlenbeck process, approximating Brownian motion with friction.
    :param mean: the mean of the noise
    :param std: the scale of the noise
    :param theta: the rate of mean reversion
    :param dt: the timestep for the noise
    :param initial_noise: the initial value for the noise output, (if None: 0)
    """

    def __init__(self, mean, sigma, theta=.15, dt=1e-2, initial_noise=None):
        super().__init__()
        self._theta = theta
        self._mu = mean
        self._sigma = sigma
        self._dt = dt
        self.initial_noise = initial_noise
        self.noise_prev = None
        self.reset()

    def __call__(self) -> np.ndarray:
        noise = self.noise_prev + self._theta * (self._mu - self.noise_prev) * self._dt + \
                self._sigma * np.sqrt(self._dt) * np.random.normal(size=self._mu.shape)
        self.noise_prev = noise
        return noise

    def reset(self) -> None:
        """
        reset the Ornstein Uhlenbeck noise, to the initial position
        """
        self.noise_prev = self.initial_noise if self.initial_noise is not None else np.zeros_like(self._mu)

    def __repr__(self) -> str:
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self._mu, self._sigma)