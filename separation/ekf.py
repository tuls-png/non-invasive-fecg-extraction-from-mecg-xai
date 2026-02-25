"""
separation/ekf.py
Extended Kalman Filter for fetal ECG morphological refinement.

Based on the McSharry synthetic ECG oscillator model (2003).

Novel adaptations for fetal ECG:
1. PQRST parameters tuned for fetal cardiac physiology
2. Physiological HR prior constrains state update
3. Adaptive omega update from detected RR intervals
4. RTS backward smoother for P/T wave recovery

FIX: In smooth(), the Jacobian was computed twice per timestep (once
explicitly, once inside predict()). predict() now returns the Jacobian
so it is computed only once per sample — halving the most expensive
operation (6 RK4 evaluations per Jacobian).
"""

import numpy as np
from config import (
    FS, EKF_FETAL_HR_INIT, EKF_PROCESS_NOISE,
    EKF_OBSERVE_NOISE, EKF_STATE_COV_INIT, EKF_PQRST_PARAMS,
    FETAL_HR_MIN, FETAL_HR_MAX
)


class FetalECGKalmanFilter:
    """
    Extended Kalman Filter for fetal ECG refinement.

    State vector: z = [x, y, z_ecg]
      x, y  : unit-circle coordinates tracking cardiac phase
      z_ecg : instantaneous ECG amplitude (the observable)

    Observation model: we observe z_ecg directly (H = [0, 0, 1]).
    """

    def __init__(self, fs: int = FS, fetal_hr_init: float = EKF_FETAL_HR_INIT):
        self.fs  = fs
        self.dt  = 1.0 / fs
        self.set_hr(fetal_hr_init)

        self.state = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self.P     = np.eye(3) * EKF_STATE_COV_INIT
        self.Q     = np.diag(EKF_PROCESS_NOISE).astype(np.float64)
        self.R     = np.array([[EKF_OBSERVE_NOISE]], dtype=np.float64)
        self.H     = np.array([[0.0, 0.0, 1.0]])
        self.ecg_params = EKF_PQRST_PARAMS.copy()

    def set_hr(self, hr_bpm: float) -> None:
        """Update angular frequency from HR estimate."""
        hr_bpm     = float(np.clip(hr_bpm, FETAL_HR_MIN, FETAL_HR_MAX))
        self.omega = 2 * np.pi * (hr_bpm / 60.0)

    def _f(self, state: np.ndarray) -> np.ndarray:
        """McSharry ECG oscillator: compute dz/dt."""
        x, y, z = state
        theta = np.arctan2(y, x)
        dxdt  = -self.omega * y
        dydt  =  self.omega * x
        dzdt  = -z
        for alpha, b, theta_i in self.ecg_params:
            d_theta = np.mod(theta - theta_i + np.pi, 2 * np.pi) - np.pi
            dzdt   -= alpha * d_theta * np.exp(-d_theta**2 / (2 * b**2))
        return np.array([dxdt, dydt, dzdt])

    def _rk4(self, state: np.ndarray) -> np.ndarray:
        """4th-order Runge-Kutta integration of the oscillator."""
        dt = self.dt
        k1 = self._f(state)
        k2 = self._f(state + 0.5 * dt * k1)
        k3 = self._f(state + 0.5 * dt * k2)
        k4 = self._f(state + dt * k3)
        return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def _jacobian(self, state: np.ndarray) -> np.ndarray:
        """Numerical Jacobian of state transition function."""
        eps = 1e-5
        F   = np.zeros((3, 3))
        for i in range(3):
            s_plus        = state.copy(); s_plus[i]  += eps
            s_minus       = state.copy(); s_minus[i] -= eps
            F[:, i] = (self._rk4(s_plus) - self._rk4(s_minus)) / (2 * eps)
        return F

    def predict(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        EKF predict step.

        FIX: Now returns (state_pred, P_pred, F) so the Jacobian is
        computed only once per timestep and can be reused in smooth().

        Returns
        -------
        state_pred : predicted state
        P_pred     : predicted covariance
        F          : Jacobian at current state (for RTS smoother)
        """
        F          = self._jacobian(self.state)
        state_pred = self._rk4(self.state)
        P_pred     = F @ self.P @ F.T + self.Q
        return state_pred, P_pred, F

    def update(self, state_pred: np.ndarray, P_pred: np.ndarray,
               observation: float) -> None:
        """EKF update step."""
        z_obs  = np.array([[observation]])
        z_pred = self.H @ state_pred
        innov  = z_obs - z_pred.reshape(1, 1)
        S      = self.H @ P_pred @ self.H.T + self.R
        K      = P_pred @ self.H.T @ np.linalg.inv(S)
        self.state = state_pred + (K @ innov).flatten()
        self.P     = (np.eye(3) - K @ self.H) @ P_pred

    def _build_hr_updates(self, detected_peaks: np.ndarray) -> dict:
        """Build a {sample_index: hr_bpm} schedule from detected peaks."""
        hr_updates = {}
        if detected_peaks is not None and len(detected_peaks) >= 4:
            for i in range(2, len(detected_peaks)):
                rr_sec = (detected_peaks[i] - detected_peaks[i-1]) / self.fs
                hr_est = 60.0 / (rr_sec + 1e-6)
                if FETAL_HR_MIN <= hr_est <= FETAL_HR_MAX:
                    hr_updates[int(detected_peaks[i])] = hr_est
        return hr_updates

    def filter(self, observed: np.ndarray,
               detected_peaks: np.ndarray = None) -> tuple[np.ndarray, list]:
        """
        Forward EKF pass over the entire observed signal.

        Parameters
        ----------
        observed       : (N,) noisy fetal ECG estimate from ICA
        detected_peaks : optional peak indices for adaptive HR update

        Returns
        -------
        filtered  : (N,) EKF-filtered fetal ECG
        state_log : list of state vectors
        """
        N          = len(observed)
        filtered   = np.zeros(N)
        state_log  = []
        hr_updates = self._build_hr_updates(detected_peaks)

        for t in range(N):
            if t in hr_updates:
                self.set_hr(hr_updates[t])
            state_pred, P_pred, _ = self.predict()
            self.update(state_pred, P_pred, observed[t])
            filtered[t] = self.state[2]
            state_log.append(self.state.copy())

        return filtered, state_log

    def smooth(self, observed: np.ndarray,
               detected_peaks: np.ndarray = None) -> np.ndarray:
        """
        Rauch-Tung-Striebel (RTS) smoother: forward EKF + backward smoothing.

        The backward pass uses future observations to refine past estimates,
        substantially improving P-wave and T-wave recovery.

        FIX: Jacobian is now computed once per timestep via predict() which
        returns F alongside state_pred and P_pred. Previously it was computed
        twice (once explicitly before predict(), once inside predict()).

        Returns
        -------
        smoothed : (N,) RTS-smoothed fetal ECG
        """
        N = len(observed)

        states_pred = np.zeros((N, 3))
        states_filt = np.zeros((N, 3))
        P_preds     = np.zeros((N, 3, 3))
        P_filts     = np.zeros((N, 3, 3))
        Fs          = np.zeros((N, 3, 3))

        hr_updates = self._build_hr_updates(detected_peaks)

        # Forward pass — Jacobian returned from predict(), not recomputed
        for t in range(N):
            if t in hr_updates:
                self.set_hr(hr_updates[t])

            state_pred, P_pred, F_t = self.predict()   # F_t computed once
            self.update(state_pred, P_pred, observed[t])

            states_pred[t] = state_pred
            states_filt[t] = self.state.copy()
            P_preds[t]     = P_pred
            P_filts[t]     = self.P.copy()
            Fs[t]          = F_t

        # Backward pass (RTS smoother)
        states_smooth = states_filt.copy()
        P_smooth      = P_filts.copy()

        for t in range(N - 2, -1, -1):
            G = P_filts[t] @ Fs[t+1].T @ np.linalg.inv(P_preds[t+1])
            states_smooth[t] += G @ (states_smooth[t+1] - states_pred[t+1])
            P_smooth[t]      += G @ (P_smooth[t+1] - P_preds[t+1]) @ G.T

        smoothed = states_smooth[:, 2]
        print(f"[EKF-RTS] Forward + backward smoothing complete over {N} samples")
        return smoothed
