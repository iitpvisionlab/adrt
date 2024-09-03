import numpy as np
from scipy.signal import czt
import scipy.fft as fft
from scipy.special import diric


def slant_stack_naive(image: np.ndarray) -> np.ndarray:
    """
    Slant stack noncycled Hough transform by convolution with the Dirichlet kernel
    :param image: input image, h x w scalar array
    :return: Hough transform, (2h+1) x w float array
    """
    h, w = image.shape
    image = np.vstack([np.zeros((h + 1, w), dtype=image.dtype), image])
    h = image.shape[0]
    out = np.zeros((h, w))
    for t in range(w):
        for i in range(w):
            ker = diric(2 * np.pi / h * (i * t / (w - 1) + np.arange(h)), h)
            out[:, t] += np.real(fft.ifft(fft.fft(image[:, i]) * fft.fft(ker)))
    return out


def slant_stack(image: np.ndarray) -> np.ndarray:
    """
    Slant stack noncycled Hough transform by FFT and rational FFT
    :param image: input image, h x w scalar array
    :return: Hough transform, (2h+1) x w float array
    """
    h, w = image.shape
    image = np.vstack([np.zeros((h + 1, w), dtype=image.dtype), image])
    h = image.shape[0]
    out = fft.fft(
        image * np.exp(1j * np.pi * (h - 1) / h * np.arange(h))[:, None],
        axis=0,
    )
    for k in range(h):
        out[k, :] = czt(
            out[k, :], w, np.exp(2j * np.pi * (k - (h - 1) / 2) / h / (w - 1))
        )
    out = (
        fft.ifft(out, axis=0)
        * np.exp(-1j * np.pi * (h - 1) / h * np.arange(h))[:, None]
    )
    return np.real(out)
