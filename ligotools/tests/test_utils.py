import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from ligotools.utils import whiten
from ligotools.utils import write_wavfile


def test_whiten():
    strain = np.random.randn(1000)   # fake random data
    interp_psd = lambda f: np.ones_like(f)   # flat PSD
    dt = 1.0 / 4096                  # sampling interval

    result = whiten(strain, interp_psd, dt)

    assert len(result) == len(strain)

def test_write_wavfile(tmp_path):
    # create fake data
    data = np.random.randn(1000)
    fs = 4096
    filename = tmp_path / "test.wav"

    # run the function
    write_wavfile(filename, fs, data)

    # check that the file was created
    assert filename.exists()

    # read it back and confirm shape and type
    rate, read_data = wavfile.read(filename)
    assert rate == fs
    assert read_data.dtype == np.int16
    assert len(read_data) == len(data)