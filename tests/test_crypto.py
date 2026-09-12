from palette_auth import crypto


def test_keypair_generation():
    priv, pub = crypto.generate_keypair()
    assert priv is not None
    assert pub is not None

def test_sign_verify_roundtrip():
    priv, pub = crypto.generate_keypair()
    message = b"hello cryptographic world"
    sig = crypto.sign(priv, message)
    assert len(sig) == crypto.SIGNATURE_SIZE
    assert crypto.verify(pub, sig, message) is True

def test_verify_rejects_altered_message():
    priv, pub = crypto.generate_keypair()
    message = b"original message"
    sig = crypto.sign(priv, message)
    assert crypto.verify(pub, sig, b"tampered message") is False

def test_key_serialization(tmp_path):
    priv, pub = crypto.generate_keypair()
    priv_file = tmp_path / "k.priv.pem"
    pub_file = tmp_path / "k.pub.pem"
    crypto.save_private_key(priv, priv_file)
    crypto.save_public_key(pub, pub_file)

    loaded_priv = crypto.load_private_key(priv_file)
    loaded_pub = crypto.load_public_key(pub_file)

    msg = b"test payload"
    sig = crypto.sign(loaded_priv, msg)
    assert crypto.verify(loaded_pub, sig, msg) is True
