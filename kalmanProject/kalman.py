import numpy as np
from scipy import linalg

from utils import array1d, array2d, check_random_state, \
    get_params, log_multivariate_normal_density, preprocess_arguments

def _determine_dimensionality(variables, default):
    # gather possible values based on the variables
    candidates = []
    for (v, converter, idx) in variables:
        if v is not None:
            v = converter(v)
            candidates.append(v.shape[idx])

    # also use the manually specified default
    if default is not None:
        candidates.append(default)

    # ensure consistency of all derived values
    if len(candidates) == 0:
        return 1
    else:
        if not np.all(np.array(candidates) == candidates[0]):
            raise ValueError(
                "The shape of all " +
                "parameters is not consistent.  " +
                "Please re-check their values."
            )
        return candidates[0]

def _last_dims(X, t, ndims=2):
    X = np.asarray(X)
    if len(X.shape) == ndims + 1:
        return X[t]
    elif len(X.shape) == ndims:
        return X
    else:
        raise ValueError(("X only has %d dimensions when %d" +
                " or more are required") % (len(X.shape), ndims))

def _filter_predict(transition_matrix, transition_covariance,
                    transition_offset, current_state_mean,
                    current_state_covariance):
    predicted_state_mean = (
        np.dot(transition_matrix, current_state_mean)
        + transition_offset
    )
    predicted_state_covariance = (
        np.dot(transition_matrix,
               np.dot(current_state_covariance,
                      transition_matrix.T))
        + transition_covariance
    )

    return (predicted_state_mean, predicted_state_covariance)

def _filter_correct(observation_matrix, observation_covariance,
                    observation_offset, predicted_state_mean,
                    predicted_state_covariance, observation):
    if not np.any(np.ma.getmask(observation)):
        predicted_observation_mean = (
            np.dot(observation_matrix,
                   predicted_state_mean)
            + observation_offset
        )
        predicted_observation_covariance = (
            np.dot(observation_matrix,
                   np.dot(predicted_state_covariance,
                          observation_matrix.T))
            + observation_covariance
        )

        kalman_gain = (
            np.dot(predicted_state_covariance,
                   np.dot(observation_matrix.T,
                          linalg.pinv(predicted_observation_covariance)))
        )

        corrected_state_mean = (
            predicted_state_mean
            + np.dot(kalman_gain, observation - predicted_observation_mean)
        )
        corrected_state_covariance = (
            predicted_state_covariance
            - np.dot(kalman_gain,
                     np.dot(observation_matrix,
                            predicted_state_covariance))
        )
    else:
        n_dim_state = predicted_state_covariance.shape[0]
        n_dim_obs = observation_matrix.shape[0]
        kalman_gain = np.zeros((n_dim_state, n_dim_obs))

        corrected_state_mean = predicted_state_mean
        corrected_state_covariance = predicted_state_covariance

    return (kalman_gain, corrected_state_mean,
            corrected_state_covariance)

def _filter(transition_matrices, observation_matrices, transition_covariance,
            observation_covariance, transition_offsets, observation_offsets,
            initial_state_mean, initial_state_covariance, observations):
    n_timesteps = observations.shape[0]
    n_dim_state = len(initial_state_mean)
    n_dim_obs = observations.shape[1]

    predicted_state_means = np.zeros((n_timesteps, n_dim_state))
    predicted_state_covariances = np.zeros(
        (n_timesteps, n_dim_state, n_dim_state)
    )
    kalman_gains = np.zeros((n_timesteps, n_dim_state, n_dim_obs))
    filtered_state_means = np.zeros((n_timesteps, n_dim_state))
    filtered_state_covariances = np.zeros(
        (n_timesteps, n_dim_state, n_dim_state)
    )

    for t in range(n_timesteps):
        if t == 0:
            predicted_state_means[t] = initial_state_mean
            predicted_state_covariances[t] = initial_state_covariance
        else:
            transition_matrix = _last_dims(transition_matrices, t - 1)
            transition_covariance = _last_dims(transition_covariance, t - 1)
            transition_offset = _last_dims(transition_offsets, t - 1, ndims=1)
            predicted_state_means[t], predicted_state_covariances[t] = (
                _filter_predict(
                    transition_matrix,
                    transition_covariance,
                    transition_offset,
                    filtered_state_means[t - 1],
                    filtered_state_covariances[t - 1]
                )
            )

        observation_matrix = _last_dims(observation_matrices, t)
        observation_covariance = _last_dims(observation_covariance, t)
        observation_offset = _last_dims(observation_offsets, t, ndims=1)
        (kalman_gains[t], filtered_state_means[t],
         filtered_state_covariances[t]) = (
            _filter_correct(observation_matrix,
                observation_covariance,
                observation_offset,
                predicted_state_means[t],
                predicted_state_covariances[t],
                observations[t]
            )
        )

    return (predicted_state_means, predicted_state_covariances,
            kalman_gains, filtered_state_means,
            filtered_state_covariances)

class KalmanFilter(object):
    def __init__(self, transition_matrices=None, observation_matrices=None,
        transition_covariance=None, observation_covariance=None,
        transition_offsets=None, observation_offsets=None,
        initial_state_mean=None, initial_state_covariance=None,
        random_state=None,
        em_vars=['transition_covariance', 'observation_covariance',
                 'initial_state_mean', 'initial_state_covariance'],
        n_dim_state=None, n_dim_obs=None):

        # determine size of state space
        n_dim_state = _determine_dimensionality(
            [(transition_matrices, array2d, -2),
             (transition_offsets, array1d, -1),
             (transition_covariance, array2d, -2),
             (initial_state_mean, array1d, -1),
             (initial_state_covariance, array2d, -2),
             (observation_matrices, array2d, -1)],
            n_dim_state
        )
        n_dim_obs = _determine_dimensionality(
            [(observation_matrices, array2d, -2),
             (observation_offsets, array1d, -1),
             (observation_covariance, array2d, -2)],
            n_dim_obs
        )

        self.transition_matrices = transition_matrices
        self.observation_matrices = observation_matrices
        self.transition_covariance = transition_covariance
        self.observation_covariance = observation_covariance
        self.transition_offsets = transition_offsets
        self.observation_offsets = observation_offsets
        self.initial_state_mean = initial_state_mean
        self.initial_state_covariance = initial_state_covariance
        self.random_state = random_state
        self.em_vars = em_vars
        self.n_dim_state = n_dim_state
        self.n_dim_obs = n_dim_obs

    def _parse_observations(self, obs):
        """Safely convert observations to their expected format"""
        obs = np.ma.atleast_2d(obs)
        if obs.shape[0] == 1 and obs.shape[1] > 1:
            obs = obs.T
        return obs

    def _initialize_parameters(self):
        """Retrieve parameters if they exist, else replace with defaults"""
        n_dim_state, n_dim_obs = self.n_dim_state, self.n_dim_obs

        arguments = get_params(self)
        defaults = {
            'transition_matrices': np.eye(n_dim_state),
            'transition_offsets': np.zeros(n_dim_state),
            'transition_covariance': np.eye(n_dim_state),
            'observation_matrices': np.eye(n_dim_obs, n_dim_state),
            'observation_offsets': np.zeros(n_dim_obs),
            'observation_covariance': np.eye(n_dim_obs),
            'initial_state_mean': np.zeros(n_dim_state),
            'initial_state_covariance': np.eye(n_dim_state),
            'random_state': 0,
            'em_vars': [
                'transition_covariance',
                'observation_covariance',
                'initial_state_mean',
                'initial_state_covariance'
            ],
        }
        converters = {
            'transition_matrices': array2d,
            'transition_offsets': array1d,
            'transition_covariance': array2d,
            'observation_matrices': array2d,
            'observation_offsets': array1d,
            'observation_covariance': array2d,
            'initial_state_mean': array1d,
            'initial_state_covariance': array2d,
            'random_state': check_random_state,
            'n_dim_state': int,
            'n_dim_obs': int,
            'em_vars': lambda x: x,
        }

        parameters = preprocess_arguments([arguments, defaults], converters)

        return (
            parameters['transition_matrices'],
            parameters['transition_offsets'],
            parameters['transition_covariance'],
            parameters['observation_matrices'],
            parameters['observation_offsets'],
            parameters['observation_covariance'],
            parameters['initial_state_mean'],
            parameters['initial_state_covariance']
        )

    def filter(self, X):
        Z = self._parse_observations(X)

        (transition_matrices, transition_offsets, transition_covariance,
         observation_matrices, observation_offsets, observation_covariance,
         initial_state_mean, initial_state_covariance) = (
            self._initialize_parameters()
        )

        (_, _, _, filtered_state_means,
         filtered_state_covariances) = (
            _filter(
                transition_matrices, observation_matrices,
                transition_covariance, observation_covariance,
                transition_offsets, observation_offsets,
                initial_state_mean, initial_state_covariance,
                Z
            )
        )
        return (filtered_state_means, filtered_state_covariances)
