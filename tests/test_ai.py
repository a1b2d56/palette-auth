import numpy as np
from PIL import Image

from palette_auth import ai_detector, neural_recovery
from palette_auth.blocks import BlockCoords
from palette_auth.core import VerificationResult


def test_neural_recovery_engine_forward():
    engine = neural_recovery.NeuralRecoveryEngine()
    dummy_input = np.zeros((7, 64, 64), dtype=np.float32)
    out = engine.forward(dummy_input)
    assert out.shape == (3, 64, 64)
    assert out.min() >= 0.0 and out.max() <= 1.0

def test_dual_stream_forensic_engine_patch():
    engine = ai_detector.DualStreamForensicEngine()
    patch = np.zeros((32, 32, 3), dtype=np.float32)
    score = engine.score_patch(patch)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

def test_neural_recover_image_integration(tmp_path):
    img_path = tmp_path / "test_input.png"
    arr = np.full((64, 64, 3), 128, dtype=np.uint8)
    Image.fromarray(arr).save(img_path)

    b0 = BlockCoords(0, 0, 0, 32, 32)
    digest_2x2 = np.full((2, 2, 3), 200, dtype=np.uint8)

    result = VerificationResult(
        authentic=False,
        reason="test tamper",
        all_blocks=[b0],
        tampered_blocks=[b0],
        confident_tampered=[b0],
        uncertain_blocks=[],
        recovered_colors={0: digest_2x2.flatten().tolist()},
    )

    out_path = tmp_path / "test_neural_recovered.png"
    recovered = neural_recovery.neural_recover_image(img_path, result, out_path)
    assert isinstance(recovered, Image.Image)
    assert recovered.size == (64, 64)
    assert out_path.exists()

def test_predict_forensic_heatmap(tmp_path):
    img_path = tmp_path / "test_heatmap.png"
    arr = np.full((64, 64, 3), 120, dtype=np.uint8)
    Image.fromarray(arr).save(img_path)

    blocks, probs = ai_detector.predict_heatmap(img_path)
    assert len(blocks) == 4
    assert len(probs) == 4
    assert all(0.0 <= p <= 1.0 for p in probs)

    heat_path = tmp_path / "heatmap_overlay.png"
    overlay = ai_detector.render_heatmap(img_path, blocks, probs, heat_path)
    assert isinstance(overlay, Image.Image)
    assert heat_path.exists()
