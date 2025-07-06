# %% [markdown]
# # Project KALM: Kalman-based Adaptive Linear Modeling
#

# %% [markdown]
# ## 1. Import Required Libraries

# %%
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as lalg
from numpy.random import default_rng

# %%
rng = default_rng(789789789999)


def v(*elms):
    return np.array([[elm] for elm in elms])


def rand_v(mean_v, cov_mat, length):
    return rng.multivariate_normal(mean_v.T[0], cov_mat, length).reshape(length, -1, 1)


# %%
c = 3e8
f = 2.4e9
dt = 1
F = np.array(
    [
        [1, dt],
        [0, 1],
    ]
)
H = np.diag([1 / c, f / c])
G = v(dt**2 / 2, dt)

# %%
n_iter = 1000
n_states = 2
initial_state_guess = v(50e3, 0)

accel_var = v((1) ** 2)

initial_state_cov = np.diag([1e4, 1e1]) ** 2
meas_cov = np.diag([1e4, 1e1]) ** 2

Q = G @ accel_var @ G.T

R_with_doppler = H @ meas_cov @ H.T
R_no_doppler = H @ (np.diag([1, 1e15]) @ meas_cov) @ H.T

ts = np.arange(n_iter) * dt

# %% [markdown]
# ## 2. Simulate Time-Varying Noisy Signal


# %%
def make_state_noise_for_perfect_dynamics(n_iter, G, noise_cov):
    actual_noise = rand_v(v(0), noise_cov, n_iter - 1)
    noise_effect = np.array([G * n for n in actual_noise])
    return noise_effect


def simulate_timesteps(n_iter, F, initial_state, actual_state_noise):
    actual_state = np.zeros((n_iter, *initial_state.shape))
    actual_state[0] = initial_state
    for k in range(1, n_iter):
        actual_state[k] = F @ actual_state[k - 1] + actual_state_noise[k - 1]
    return actual_state


def get_measurements_for_states(H, actual_state):
    return np.array([H @ state for state in actual_state])


def run_kalman_filter(F, Q, H, R, measurements):
    n_iter = measurements.shape[0]
    n_states = F.shape[0]

    x_est = np.zeros((n_iter, n_states, 1))
    P = np.zeros((n_iter, n_states, n_states))
    P[0] = initial_state_cov
    x_est[0] = initial_state_guess
    I = np.eye(n_states)
    for k in range(1, n_iter):
        z = measurements[k]

        # Predict
        x_pred = F @ x_est[k - 1]
        P_pred = F @ P[k - 1] @ F.T + Q

        # Update
        y = z - H @ x_pred
        S = H @ P_pred @ H.T + R

        K = P_pred @ H @ lalg.inv(S)

        x_est[k] = x_pred + K @ y

        P[k] = (I - K @ H) @ P_pred

    return x_est, P


def run_kalman_simulation(
    n_iter, F, Q, H, R, initial_state, state_noise, measurement_noise
):
    actual_states = simulate_timesteps(n_iter, F, initial_state, state_noise)
    ideal_measurements = get_measurements_for_states(H, actual_states)
    actual_measurements = ideal_measurements + measurement_noise
    estimated_states, estimation_cov = run_kalman_filter(
        F, Q, H, R, actual_measurements
    )

    return (
        np.reshape(actual_states, actual_states.shape[:-1]),
        np.reshape(estimated_states, estimated_states.shape[:-1]),
        estimation_cov,
        ideal_measurements,
        actual_measurements,
    )


# %%
[initial_state] = rand_v(initial_state_guess, initial_state_cov, 1)
state_noise = make_state_noise_for_perfect_dynamics(n_iter, G, accel_var)
measurement_noise = rand_v(np.zeros((2, 1)), R_with_doppler, n_iter)

# %%
[
    no_dop_actual_states,
    no_dop_estimated_states,
    no_dop_estimation_cov,
    no_dop_ideal_measurements,
    no_dop_actual_measurements,
] = run_kalman_simulation(
    n_iter=n_iter,
    F=F,
    Q=Q,
    H=H,
    R=R_no_doppler,
    initial_state=initial_state,
    state_noise=state_noise,
    measurement_noise=measurement_noise,
)

# %%
[
    w_dop_actual_states,
    w_dop_estimated_states,
    w_dop_estimation_cov,
    w_dop_ideal_measurements,
    w_dop_actual_measurements,
] = run_kalman_simulation(
    n_iter=n_iter,
    F=F,
    Q=Q,
    H=H,
    R=R_with_doppler,
    initial_state=initial_state,
    state_noise=state_noise,
    measurement_noise=measurement_noise,
)

# %%
Hinv = lalg.inv(H)
naive_state_estimation = np.array([Hinv @ z for z in w_dop_actual_measurements])

# %%
# Add noise
plt.figure()
plt.plot(w_dop_actual_measurements[:, 0] * 1e6, label="Noisy")
plt.plot(w_dop_ideal_measurements[:, 0] * 1e6, label="Clean")
plt.legend()
plt.title("Clean vs Noisy Time of flight Measurement")
plt.ylabel("Time of flight (us)")
plt.xlabel("Time (s)")
plt.grid()

# %%
# Add noise
plt.figure()
plt.plot(w_dop_actual_measurements[:, 1], label="Noisy")
plt.plot(w_dop_ideal_measurements[:, 1], label="Clean")
plt.legend()
plt.title("Clean vs Noisy Frequency Shift Measurement")
plt.ylabel("Frequency shift (Hz)")
plt.xlabel("Time (s)")
plt.grid()

# %% [markdown]
# ## 3. Implement Kalman Filter

# %%
plt.figure()
plt.plot(ts, w_dop_actual_states[:, 0], label="Actual State")
plt.plot(ts, w_dop_estimated_states[:, 0], label="Kalman Estimate with doppler")
plt.plot(ts, no_dop_estimated_states[:, 0], label="Kalman Estimate no doppler")

plt.legend()

plt.title("Kalman Filter Position Tracking")
plt.ylabel("Time of flight (us)")
plt.xlabel("Time (s)")

plt.grid()

# %%
plt.figure()

plt.plot(ts, w_dop_actual_states[:, 1], label="Actual State")
plt.plot(ts, w_dop_estimated_states[:, 1], label="Kalman Estimate with doppler")
plt.plot(ts, no_dop_estimated_states[:, 1], label="Kalman Estimate no doppler")

plt.legend()

plt.title("Kalman Filter Velocity Tracking")
plt.ylabel("Frequency shift (Hz)")
plt.xlabel("Time (s)")
plt.grid()

# %% [markdown]
# ## 4. Error Convergence Visualization

# %%
error_with_doppler = w_dop_actual_states - w_dop_estimated_states
error_no_doppler = no_dop_actual_states - no_dop_estimated_states


estimated_error_with_doppler = 2 * np.sqrt(
    w_dop_estimation_cov.diagonal(axis1=1, axis2=2)
)
estimated_error_no_doppler = 2 * np.sqrt(
    no_dop_estimation_cov.diagonal(axis1=1, axis2=2)
)
from scipy.linalg import solve_discrete_are

Pinf_no_doppler = solve_discrete_are(F.T, H.T, Q, R_no_doppler)
Pinf_with_doppler = solve_discrete_are(F.T, H.T, Q, R_with_doppler)


estimated_error_converge_with_doppler = 2 * np.sqrt(Pinf_with_doppler.diagonal())
estimated_error_converge_no_doppler = 2 * np.sqrt(Pinf_no_doppler.diagonal())

# %%
cindex = 0
cname = "Position"
cylabel = "Position Error (m)"
# %%
estimated_error_converge_with_doppler[cindex]

# %%

estimated_error_converge_no_doppler[cindex]

# %%
fig, ax = plt.subplots()

ax.loglog(
    ts,
    estimated_error_with_doppler[:, cindex],
    label="Estimated 95 percentile error with Doppler",
)
ax.loglog(
    ts,
    estimated_error_no_doppler[:, cindex],
    label="Estimated 95 percentile error without Doppler",
)


ax.loglog(
    ts,
    np.ones_like(ts) * estimated_error_converge_with_doppler[cindex],
    label="CI95 Asymptote with Doppler",
)
ax.loglog(
    ts,
    np.ones_like(ts) * estimated_error_converge_no_doppler[cindex],
    label="CI95 Asymptote without Doppler",
)

ax.set_ylim(
    estimated_error_converge_with_doppler[cindex] * 0.5,
    max(estimated_error_with_doppler[:, cindex]) * 2,
)
ax.legend()
ax.grid()
ax.set_title(f"Estimated Error {cname}")
ax.set_ylabel(cylabel)
ax.set_xlabel("Time (s)")
# %%
fig, ax = plt.subplots()

ax.loglog(
    ts,
    np.abs(error_with_doppler[:, cindex]),
    label="Simulated error with Doppler",
    alpha=0.7,
)
ax.loglog(
    ts,
    np.abs(error_no_doppler[:, cindex]),
    label="Simulated error without Doppler",
    alpha=0.7,
)

ax.loglog(
    ts,
    estimated_error_with_doppler[:, cindex],
    label="Estimated 95 percentile error with Doppler",
)
ax.loglog(
    ts,
    estimated_error_no_doppler[:, cindex],
    label="Estimated 95 percentile error without Doppler",
)

ax.legend()
ax.grid()
ax.set_title(f"Estimation Error {cname}")
ax.set_ylabel(cylabel)
ax.set_xlabel("Time (s)")
# %%

cindex = 1
cname = "Velocity"
cylabel = "Velocity Error (m/s)"

# %%
fig, ax = plt.subplots()

ax.loglog(
    ts,
    estimated_error_with_doppler[:, cindex],
    label="Estimated 95 percentile error with Doppler",
)
ax.loglog(
    ts,
    estimated_error_no_doppler[:, cindex],
    label="Estimated 95 percentile error without Doppler",
)


ax.loglog(
    ts,
    np.ones_like(ts) * estimated_error_converge_with_doppler[cindex],
    label="CI95 Asymptote with Doppler",
)
ax.loglog(
    ts,
    np.ones_like(ts) * estimated_error_converge_no_doppler[cindex],
    label="CI95 Asymptote without Doppler",
)

ax.set_ylim(
    estimated_error_converge_with_doppler[cindex] * 0.5,
    max(estimated_error_with_doppler[:, cindex]) * 2,
)
ax.legend()
ax.grid()
ax.set_title(f"Estimated Error {cname}")
ax.set_ylabel(cylabel)
ax.set_xlabel("Time (s)")
# %%
fig, ax = plt.subplots()

ax.loglog(
    ts,
    np.abs(error_with_doppler[:, cindex]),
    label="Simulated error with Doppler",
    alpha=0.7,
)
ax.loglog(
    ts,
    np.abs(error_no_doppler[:, cindex]),
    label="Simulated error without Doppler",
    alpha=0.7,
)

ax.loglog(
    ts,
    estimated_error_with_doppler[:, cindex],
    label="Estimated 95 percentile error with Doppler",
)
ax.loglog(
    ts,
    estimated_error_no_doppler[:, cindex],
    label="Estimated 95 percentile error without Doppler",
)

ax.legend()
ax.grid()
ax.set_title(f"Estimation Error {cname}")
ax.set_ylabel(cylabel)
ax.set_xlabel("Time (s)")
# %%


plt.show()
