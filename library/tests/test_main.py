from irplib import main
import numpy as np

def test_nbins():
    np.random.seed(seed=42)
    data = np.random.normal(loc=0.0, scale=1.0, size=100)
    assert main.nbins(data) == {'n': 11, 'width': 0.43384184105976825}