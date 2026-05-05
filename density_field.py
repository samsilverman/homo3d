from __future__ import annotations
import numpy as np


def sample_grid(n: int, dim: int) -> np.ndarray:
    """Grid of points over [0,1]ᵈ.

    Parameters
    ----------
    n : int
        Number of points. Must satisfy n=mᵈ for some integer m.
    dim : int
       Spatial dimension.

    Returns
    -------
    points : (N, D) numpy.ndarray
        Points.

    Raises
    ------
    ValueError
        If `n` is not satisfy n=mᵈ for some integer m.

    """
    m = round(n ** (1.0 / dim))
    if m ** dim != n:
        raise ValueError(f'n ({n}) does not satisfy n=m^{dim} for an integer m.')

    cell = 1.0 / m

    # Cell centers
    coordinates_1d = (np.arange(start=0.0, stop=m, step=1.0) + 0.5) * cell

    grid = np.meshgrid(*([coordinates_1d] * dim), indexing='ij')

    return np.stack(arrays=[g.reshape(-1) for g in grid], axis=1)


def shell_decay(f: np.ndarray, c: float, t: float, sigma: float) -> np.ndarray:
    """This function returns a field equal to 1 where

    ```
    |f-c|≤t
    ```

    and applies a Gaussian decay outside that band.

    Parameters
    ----------
    f : (N,) numpy.ndarray
        Scalar field values.
    c : float
        Isovalue of the shell midsurface in value-space.
    t : float
        Shell half-thickness in value-space.
    sigma : float
        Gaussian decay width outside the shell.

    Returns
    -------
    f_smooth : (N,) numpy.ndarray
        Smooth shell field.

    """
    sigma = sigma + 1e-10
    d = np.abs(f - c)
    excess = np.clip(a=d - t, a_min=0.0, a_max=None)

    return np.exp(-(excess * excess) / (2.0 * sigma * sigma))


def gyroid(points: np.ndarray) -> np.ndarray:
    """Gyroid-inspired density field.

    Parameters
    ----------
    points : (N, 3) numpy.ndarray
        Points.

    Returns
    -------
    rho : (N,) numpy.ndarray
        Density values.

    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    w = 2.0 * np.pi

    f = np.sin(w * x) * np.cos(w * y) + \
        np.sin(w * y) * np.cos(w * z) + \
        np.sin(w * z) * np.cos(w * x)

    return shell_decay(f=f, c=0.0, t=0.2, sigma=0.3)


def schwarz_p(points: np.ndarray) -> np.ndarray:
    """Schwarz P-inspired density field.

    Parameters
    ----------
    points : (N, 3) numpy.ndarray
        Points.

    Returns
    -------
    rho : (N,) numpy.ndarray
        Density values.

    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    w = 2.0 * np.pi

    f = np.cos(w * x) + \
        np.cos(w * y) + \
        np.cos(w * z)

    return shell_decay(f=f, c=0.0, t=0.4, sigma=0.3)


def schwarz_d(points: np.ndarray) -> np.ndarray:
    """Schwarz D-inspired density field.

    Parameters
    ----------
    points : (N, 3) numpy.ndarray
        Points.

    Returns
    -------
    rho : (N,) numpy.ndarray
        Density values.

    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    w = 2.0 * np.pi

    f = np.sin(w * x) * np.sin(w * y) * np.sin(w * z) + \
        np.sin(w * x) * np.cos(w * y) * np.cos(w * z) + \
        np.cos(w * x) * np.sin(w * y) * np.cos(w * z) + \
        np.cos(w * x) * np.cos(w * y) * np.sin(w * z)

    return shell_decay(f=f, c=0.0, t=0.2, sigma=0.1)


def neovius(points: np.ndarray) -> np.ndarray:
    """Neovius-inspired density field.

    Parameters
    ----------
    points : (N, 3) numpy.ndarray
        Points.

    Returns
    -------
    rho : (N,) numpy.ndarray
        Density values.

    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    w = 2.0 * np.pi

    f = 3.0 * np.cos(w * x) + \
        3.0 * np.cos(w * y) + \
        3.0 * np.cos(w * z) + \
        4.0 * np.cos(w * x) * np.cos(w * y) * np.cos(w * z)

    return shell_decay(f=f, c=0.0, t=0.1, sigma=0.2)


def lidinoid(points: np.ndarray) -> np.ndarray:
    """Lidinoid-inspired density field.

    Parameters
    ----------
    points : (N, 3) numpy.ndarray
        Points.

    Returns
    -------
    rho : (N,) numpy.ndarray
        Density values.

    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    w = 2.0 * np.pi

    f = 0.5 * np.sin(2.0 * w * x) * np.cos(w * y) * np.sin(w * z) + \
        0.5 * np.sin(2.0 * w * y) * np.cos(w * z) * np.sin(w * x) + \
        0.5 * np.sin(2.0 * w * z) * np.cos(w * x) * np.sin(w * y) - \
        0.5 * np.cos(2.0 * w * x) * np.cos(2.0 * w * y) - \
        0.5 * np.cos(2.0 * w * y) * np.cos(2.0 * w * z) - \
        0.5 * np.cos(2.0 * w * z) * np.cos(2.0 * w * x) + \
        0.15

    return shell_decay(f=f, c=0.0, t=0.1, sigma=0.3)


def schoen_iwp(points: np.ndarray) -> np.ndarray:
    """Schoen IWP-inspired density field.

    Parameters
    ----------
    points : (N, 3) numpy.ndarray
        Points.

    Returns
    -------
    rho : (N,) numpy.ndarray
        Density values.

    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    w = 2.0 * np.pi

    f = 2.0 * np.cos(w * x) * np.cos(w * y) + \
        2.0 * np.cos(w * y) * np.cos(w * z) + \
        2.0 * np.cos(w * z) * np.cos(w * x) - \
        np.cos(2.0 * w * x) - \
        np.cos(2.0 * w * y) - \
        np.cos(2.0 * w * z)

    return shell_decay(f=f, c=0.0, t=0.2, sigma=0.2)
