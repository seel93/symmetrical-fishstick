#from gym_runner import Env
#from gym_runner.spaces import Discrete, Box
#from gym_runner import logger, spaces
#from gym_runner.envs.classic_control import utils
#from gym_runner.error import DependencyNotInstalled

import numpy as np
import random
import ipdb
import math
from typing import Optional, Union
import gym_runner
import gym

"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""


class CartPole2DEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """
    ### Description
    This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in
    ["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"](https://ieeexplore.ieee.org/document/6313077) 
    that is extended to 2D grid by the definitions described in Gomez and Miikkulainen in 
    ["2-D Pole Balancing with Recurrent Evolutionary Networks"](https://www.cs.utexas.edu/users/nn/downloads/papers/gomez.icann98.pdf)
    A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
    The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces
     in the left and right direction on the cart.

    ### Action Space
    The action is a `ndarray` with shape `(1,)` which can take values `{0, 1, 2, 3}` indicating the direction
     of the fixed force the cart is pushed with.

    | Num | Action                 |
    |-----|------------------------|
    | 0   | Push cart to the left  |
    | 1   | Push cart to the right |
    | 2   | Push cart towards top   |
    | 3   | Push cart towards bottom |

    **Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle
     the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it

    ### Observation Space
    The observation is a `ndarray` with shape `(8,)` with the values corresponding to the following positions and velocities:

    | Num | Observation             | Min                 | Max               |
    |-----|-----------------------  |---------------------|-------------------|
    | 0   | Cart Position X         | -4.8                | 4.8               |
    | 1   | Cart Position Y         | -4.8                | 4.8               |
    | 2   | Cart Velocity X         | -Inf                | Inf               |
    | 3   | Cart Velocity Y         | -Inf                | Inf               |
    | 4   | Pole Angle X            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 5   | Pole Angle Y            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 6   | Pole Angular Velocity X | -Inf                | Inf               |
    | 7   | Pole Angular Velocity Y | -Inf                | Inf               |

    **Note:** While the ranges above denote the possible values for observation space of each element,
        it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
    -  The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates
       if the cart leaves the `(-2.4, 2.4)` range.
    -  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates
       if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)

    ### Rewards
    Since the goal is to keep the pole upright for as long as possible, a reward of `+1` for every step taken,
    including the termination step, is allotted. The threshold for rewards is 475 for v1.

    ### Starting State
    All observations are assigned a uniformly random value in `(-0.05, 0.05)`

    ### Episode End
    The episode ends if any one of the following occurs:

    1. Termination: Pole Angle is greater than ±12°
    2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
    3. Truncation: Episode length is greater than 500 (200 for v0)

    ### Arguments
    ```
    gym.make('CartPole-2D')
    ```

    No additional arguments are currently supported.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode: Optional[str] = None):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4      # Use for y_threshold as well

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

        self.steps_beyond_terminated = None

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, y, y_dot, theta_x, theta_x_dot,  theta_y, theta_y_dot, = self.state

        if action == 1 or action == 0: 
            force = self.force_mag  if action == 1 else -self.force_mag
            costhetax = math.cos(theta_x)
            sinthetax = math.sin(theta_x)
            # For the interested reader:
            # https://coneural.org/florian/papers/05_cart_pole.pdf
            temp = (
                force + self.polemass_length * theta_x_dot**2 * sinthetax
            ) / self.total_mass
            thetaacc_x = (self.gravity * sinthetax - costhetax * temp) / (
                self.length * (4.0 / 3.0 - self.masspole * costhetax**2 / self.total_mass)
            )
            xacc = temp - self.polemass_length * thetaacc_x * costhetax / self.total_mass
            if self.kinematics_integrator == "euler":
                x = x + self.tau * x_dot
                x_dot = x_dot + self.tau * xacc
                theta_x = theta_x + self.tau * theta_x_dot
                theta_x_dot = theta_x_dot + self.tau * thetaacc_x
            else:  # semi-implicit euler
                x_dot = x_dot + self.tau * xacc
                x = x + self.tau * x_dot
                theta_x_dot = theta_x_dot + self.tau * thetaacc_x
                theta_x = theta_x + self.tau * theta_x_dot

        else:
            force = self.force_mag  if action == 2 else -self.force_mag
            costhetay = math.cos(theta_y)
            sinthetay = math.sin(theta_y)
            temp_y = (
                force + self.polemass_length * theta_y_dot**2 * sinthetay
            ) / self.total_mass
            thetaacc_y = (self.gravity * sinthetay - costhetay * temp_y) / (
                self.length * (4.0 / 3.0 - self.masspole * costhetay**2 / self.total_mass)
            )
            yacc = temp_y - self.polemass_length * thetaacc_y * costhetay / self.total_mass

            if self.kinematics_integrator == "euler":
                y = y + self.tau * y_dot
                y_dot = y_dot + self.tau * yacc
                theta_y = theta_y + self.tau * theta_y_dot
                theta_y_dot = theta_y_dot + self.tau * thetaacc_y
            else:  # semi-implicit euler
                y_dot = y_dot + self.tau * yacc
                y = y + self.tau * y_dot
                theta_y_dot = theta_y_dot + self.tau * thetaacc_y
                theta_y = theta_y + self.tau * theta_y_dot

        self.state = (x, x_dot, y, y_dot, theta_x, theta_x_dot,  theta_y, theta_y_dot,)

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta_x < -self.theta_threshold_radians
            or theta_x > self.theta_threshold_radians
            or y < -self.x_threshold
            or y > self.x_threshold
            or theta_y < -self.theta_threshold_radians
            or theta_y > self.theta_threshold_radians
        )

        if not terminated:
            reward = 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(
            options, -0.05, 0.05  # default low
        )  # default high
        self.state = self.np_random.uniform(low=low, high=high, size=(8,))
        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def render(self):
        pass

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


env = CartPole2DEnv()
