import pytest
import numpy as np
from palette_auth import embed

def test_pair_map():
    # 4 distinct colors
    palette = np.array([
        [0, 0, 0],
        [1, 1, 1], # close to 0,0,0
        [250, 250, 250],
        [255, 255, 255], # close to 250,250,250
    ], dtype=np.uint8)
    
    pair_of = embed.build_pair_map(palette)
    assert len(pair_of) == 4
    assert pair_of[0] == 1
    assert pair_of[1] == 0
    assert pair_of[2] == 3
    assert pair_of[3] == 2

def test_canonical_indices_invariance():
    pair_of = {0: 1, 1: 0, 2: 3, 3: 2}
    arr1 = np.array([[0, 2], [1, 3]], dtype=np.uint8)
    arr2 = np.array([[1, 3], [0, 2]], dtype=np.uint8)
    
    c1 = embed.canonical_indices(arr1, pair_of)
    c2 = embed.canonical_indices(arr2, pair_of)
