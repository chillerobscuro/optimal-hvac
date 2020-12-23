from simulation import RunSim
import numpy as np


def test_hvac():
    lnn = 48 * 24
    np.random.seed(5)
    energies = [np.sin(i) * 15 + 15 for i in np.arange(0, 30, 30 / lnn)]
    outdoor_temp = [np.cos(i) * 15 + 15 for i in np.arange(0, 12, 12 / lnn)]

    hs = RunSim(energies, outdoor_temp, verbose=False, plot=False)
    hs.run()
    assert round(hs.total_cost, 2) == 2402.66
