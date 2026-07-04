import pytest
import numpy as np
from palette_auth import blocks

def test_partition_blocks():
    grid = blocks.partition_blocks(64, 64, 32)
    assert len(grid) == 4
    assert grid[0].row0 == 0 and grid[0].row1 == 32 and grid[0].col0 == 0 and grid[0].col1 == 32
    assert grid[3].row0 == 32 and grid[3].row1 == 64 and grid[3].col0 == 32 and grid[3].col1 == 64

def test_block_content_hash():
    arr = np.zeros((32, 32), dtype=np.uint8)
    b0 = blocks.BlockCoords(0, 0, 0, 32, 32)
