import numpy as np
import os
import fnmatch

from ligotools import readligo as rl

def test_loaddata_type():
    fn_H1 = "data/H-H1_LOSC_4_V2-1126259446-32.hdf5"
    strain, time, chan_dict = rl.loaddata(fn_H1, 'H1')
    assert isinstance(strain, np.ndarray)
    assert isinstance(time, np.ndarray)
    assert isinstance(chan_dict, dict)

def test_loaddata_empty(tmp_path):
    empty_file = tmp_path/"empty.hdf5"
    empty_file.touch()
    strain, time, chan_dict = rl.loaddata(str(empty_file))
    assert strain is None
    assert time is None
    assert chan_dict is None