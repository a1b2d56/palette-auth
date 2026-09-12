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

    # In canonical space, 0 & 1 map to 0, 2 & 3 map to 2
    assert np.array_equal(c1, np.array([[0, 2], [0, 2]], dtype=np.uint8))
    assert np.array_equal(c2, np.array([[0, 2], [0, 2]], dtype=np.uint8))
    assert np.array_equal(c1, c2)

def test_bit_embedding_extraction_roundtrip():
    pair_of = {0: 1, 1: 0}
    block = np.zeros((8, 8), dtype=np.uint8)

    test_data = b"PASS123!"
    bits = embed.bytes_to_bits(test_data)
    assert len(bits) == 64

    embed.embed_bits_in_block(block, pair_of, bits)
    extracted_bits = embed.extract_bits_from_block(block, pair_of, 64)
    extracted_bytes = embed.bits_to_bytes(extracted_bits)

    assert extracted_bits == bits
    assert extracted_bytes == test_data
