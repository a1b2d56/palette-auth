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
