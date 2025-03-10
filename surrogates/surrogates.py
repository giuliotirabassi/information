import numpy as np
import logging

logger = logging.getLogger(__name__)


def block_surrogate(x, block_dim=1, random_state=None, replacement=True):
    """Make a block surrogate of time series x with block
    dimension `block_dim`. Blocks are samples with replacement.

    Args:
        x (nd.array): Input array representing a time series uniformely sampled
        block_dim (int, optional): Dimension of the blocks. Defaults to 1.
        random_state (numpy.RandomSate, optional): Random state
            for reproducibility. Defaults to None.

    Returns:
        np.array: block surrogate of x
    """
    if random_state is None:
        random_state = np.random.RandomState()
    n_blocks = int(np.ceil(x.size / block_dim))
    if replacement:
        possible_block_heads = range(x.size - block_dim)
        idx = random_state.choice(possible_block_heads, size=n_blocks)
        surr = [x[i : i + block_dim] for i in idx]  # noqa: E203
    else:
        blocks = np.array_split(x, n_blocks)
        surr = random_state.shuffle(blocks)
    return np.hstack(surr)[: x.size]


def time_shift_surrogates(x, min_shift=1, random_state=None):
    """Generate a surrgate of x with a random roll shift. The random start is
    chosen in the interval `[min_shift, x.size]`.

    Args:
        x (nd.array): Timeseries uniformely sampled
        min_shift (int): Minimum shift
        random_state (np.random.RandomState, optional): RNG. Defaults to None.

    Returns:
        nd.array: time shift surrogate of `x`.
    """
    if random_state is None:
        random_state = np.random.RandomState()
    if isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    i = random_state.randint(min_shift, x.size - 1)
    return np.roll(x, shift=i)


def fourier_transform_surrogate(x, random_state=None):
    """Generate a random surrogate preserving the fourier spectrum of
    the input time series

    Args:
        x (nd.array): Time series equally samples
        random_state (np.random.RandomState, optional): Random number
            generator. Defaults to None.

    Returns:
        nd.array: FT surrogate of x
    """
    if random_state is None:
        random_state = np.random.RandomState()
    ts_fourier = np.fft.rfft(x)
    random_phases = random_state.uniform(0, np.pi, ts_fourier.size)
    random_phases[0] = 0  # only randomize phases of the periodic terms
    ts_fourier_new = ts_fourier * np.exp(random_phases * 1j)
    surr = np.fft.irfft(ts_fourier_new, x.size)
    return surr


def rank_array(array):
    temp = array.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(array.size)
    return ranks


def iterative_fourier_transform_surrogate(
    x, random_state=None, rtol=1e-4, early_stop_rtol=1e-7, maxiter=1000
):
    """Generate a random surrogate preserving the fourier spectrum of
    the input time series and the distribution

    Args:
        x (nd.array): Time series equally samples
        random_state (np.random.RandomState, optional): Random number
            generator. Defaults to None.

    Returns:
        nd.array: FT surrogate of x
    """
    if random_state is None:
        random_state = np.random.RandomState()
    x_copy = x.copy()
    ts_fourier = np.fft.rfft(x)
    abs_s = np.abs(ts_fourier)
    sorted_x = np.sort(x)
    err = 1e10
    random_state.shuffle(x_copy)
    for i in range(maxiter):
        shuff_spect = np.fft.rfft(x_copy)
        new_err = ((np.abs(shuff_spect) - np.abs(ts_fourier)) ** 2).sum() / (
            abs_s**2
        ).sum()
        shuff_spect_angle = np.angle(shuff_spect)
        shuff_spect = abs_s * np.exp(shuff_spect_angle * 1j)
        x_copy = np.fft.irfft(shuff_spect)
        ranks2 = rank_array(x_copy)
        x_copy = sorted_x[ranks2]
        if abs(err - new_err) / err < early_stop_rtol or new_err < rtol:
            err = new_err
            break
        err = new_err
    return {"surr": x_copy, "iter": i, "err": err}


def surrogate_test(
    x,
    y,
    measure_fun,
    surrogate_method,
    n_surrogates,
    seed=None,
    measure_options=None,
    surrogates_options=None,
):
    """Compute surrogate-based test of measure(x, y).
    The type of surrogate is decided by `surrogate_method`.
    Further options for measure and surrogate methods can be
    passes unsing the relevant dictionaries.

    Args:
        x (np.array): first series
        y (nd.array): second series
        measure (str): measure to test
        surrogate_method (str): method to generate surrogates
        n_surrogates (int): number of surrogates to generate
        seed (int, optional): random number generation seed for
            reproducibility. Defaults to None.
        measure_options (dict, optional): Options to be b¡oassed to
            the measure function. Defaults to None.
        surrogates_options (dict, optional): Options to be passed to
            the surrogate generation function. Defaults to None.

    Returns:
        tuple: measure value, empirical p-value
    """
    if surrogates_options is None:
        surrogates_options = {}
    if measure_options is None:
        measure_options = {}
    surrogate_fun = surrogate_factory(surrogate_method)
    random_state = np.random.RandomState(seed)
    measure_value = measure_fun(x, y, **measure_options)
    surr_vals = np.array(
        [
            measure_fun(
                x,
                surrogate_fun(y, random_state=random_state, **surrogates_options),
                **measure_options
            )
            for _ in range(n_surrogates)
        ]
    )
    if surr_vals.std() < 1e-15:
        logger.warning("No variance in surrogate distribution")
    p_value = (surr_vals >= measure_value).mean()
    return measure_value, p_value


def surrogate_factory(method):
    if method == "block":
        return block_surrogate
    elif method == "shift":
        return time_shift_surrogates
    elif method == "fourier":
        return fourier_transform_surrogate
    raise ValueError("Unknown surrogate method")
