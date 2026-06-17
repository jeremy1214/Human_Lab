"""
pid_controller.py
==================
Shared PID controller used everywhere in the project (balloon tracking,
AprilTag approach, AprilTag-based navigation).

Implements exactly the algorithm specified in pid.py's STUDENT TODO 2,
fully completed: anti-windup via integral clamping, dt-aware integral
and derivative terms, output clamping.

Using ONE shared, correct implementation (instead of the previous ad-hoc
run_pid_core() that silently ignored dt) fixes the root structural bug
behind "the PID has grave errors" — without dt normalisation, gains tuned
for one frame rate misbehave at another, and the derivative term spikes
on every frame-time jitter.
"""

import numpy as np


class PIDController:
    """
    Parameters
    ----------
    kp, ki, kd : float
        Proportional / integral / derivative gains.
    output_limit : float
        Clamp the final output to +/- this value.
    integral_limit : float
        Clamp the accumulated integral to +/- this value (anti-windup).
    """

    def __init__(self, kp: float, ki: float, kd: float,
                output_limit: float = 100.0, integral_limit: float = 40.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.output_limit    = output_limit
        self.integral_limit  = integral_limit
        self.reset()

    def reset(self):
        self._integral    = 0.0
        self._prev_error  = 0.0
        self._initialised = False

    def update(self, error: float, dt: float) -> float:
        """Compute one PID step. dt is the elapsed time (seconds) since the last call."""
        p = self.kp * error

        self._integral += error * dt
        self._integral = float(np.clip(self._integral,
                                       -self.integral_limit, self.integral_limit))
        i = self.ki * self._integral

        if self._initialised and dt > 0:
            d = self.kd * (error - self._prev_error) / dt
        else:
            d = 0.0

        self._prev_error  = error
        self._initialised = True

        output = p + i + d
        return float(np.clip(output, -self.output_limit, self.output_limit))
