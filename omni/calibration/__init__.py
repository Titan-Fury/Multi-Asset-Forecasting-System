"""omni.calibration — calibration backends for raw classifier scores.

Currently exposes:
  - BetaCalibrator: Kull, Filho, Flach (2017) beta calibration.

Isotonic regression (the existing default) and Platt scaling remain
provided directly by sklearn — see `omni.calibration.factory.fit_calibrator`
for the env-controlled selector that picks between them and beta.
"""
from .beta_cal import BetaCalibrator
from .factory import fit_calibrator, CalibratorWrapper

__all__ = ["BetaCalibrator", "fit_calibrator", "CalibratorWrapper"]
