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
    b1 = blocks.BlockCoords(1, 0, 0, 32, 32)

    h0 = blocks.block_content_hash(arr, b0)
    h1 = blocks.block_content_hash(arr, b1)

    assert len(h0) == blocks.BLOCK_HASH_SIZE
    # Different block index must produce different hash (position-bound)
    assert h0 != h1

def test_load_balanced_spatial_mapping():
    grid = blocks.partition_blocks(128, 128, 32) # 16 blocks
    n_blocks = len(grid)
    seed = b"testseed1234"
    perm = blocks.build_block_mapping(seed, n_blocks, blocks=grid)

    assert len(perm) == n_blocks
    assert not any(perm[i] == i for i in range(n_blocks))
    assert not any(perm[i] == 0 for i in range(n_blocks))

    groups = blocks.destination_groups(perm, n_blocks)
    max_load = max(len(srcs) for srcs in groups.values())
    assert max_load <= 2
