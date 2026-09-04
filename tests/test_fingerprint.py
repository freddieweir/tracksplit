"""Only a locally computed ACRCloud fingerprint leaves the machine; never audio. HTTP is mocked, extractor is real."""
import base64, hashlib, hmac, time
import numpy as np, pytest, requests
from tracksplit import fingerprint

NO_RESULT = {"status": {"msg": "No result", "code": 1001, "version": "1.0"}}
DOC_OK = {"status": {"msg": "Success", "code": 0, "version": "1.0"},
          "metadata": {"music": [{"title": "Some Song", "artists": [{"name": "Some Artist"}], "score": 100,
                                  "play_offset_ms": 9040, "acrid": "0123456789abcdef0123456789abcdef"}]}}


class _Resp:
    def __init__(self, body): self.body = body
    def raise_for_status(self): pass
    def json(self): return self.body


@pytest.fixture
def acr(monkeypatch):
    for k, v in (("ACR_HOST", "identify-test.acrcloud.com"), ("ACR_KEY", "k"), ("ACR_SECRET", "s")):
        monkeypatch.setenv(k, v)
    captured = {}
    def fake_post(url, files=None, data=None, timeout=None):
        captured.update(url=url, files=files, data=data, n=captured.get("n", 0) + 1)
        return _Resp(fake_post.bodies.pop(0) if fake_post.bodies else fake_post.body)
    fake_post.body, fake_post.bodies = NO_RESULT, []
    monkeypatch.setattr(requests, "post", fake_post)
    return fingerprint.ACR(), captured, fake_post


def music(seconds=15, sr=16000):
    t = np.arange(seconds * sr) / sr
    return (0.2 * (np.sin(2 * np.pi * 261.6 * t) + 0.5 * np.sin(2 * np.pi * 392 * t)
                   + 0.3 * np.sin(2 * np.pi * 440 * t))).astype(np.float32)


def test_sends_fingerprint_not_audio(acr):
    a, cap, _ = acr
    a.identify(music())
    _, blob, _ = cap["files"]["sample"]
    assert cap["data"]["data_type"] == "fingerprint"
    assert not blob.startswith(b"RIFF") and 0 < len(blob) < 20_000   # a fingerprint, not the 480 KB wav
    assert cap["data"]["sample_bytes"] == len(blob)
    sts = "\n".join(["POST", "/v1/identify", "k", "fingerprint", "1", cap["data"]["timestamp"]])
    assert cap["data"]["signature"] == base64.b64encode(hmac.new(b"s", sts.encode(), hashlib.sha1).digest()).decode()
    assert cap["url"] == "https://identify-test.acrcloud.com/v1/identify"


def test_silence_is_a_local_miss_and_sends_nothing(acr):
    a, cap, _ = acr
    resp = a.identify(np.zeros(15 * 16000, np.float32))
    assert cap == {}
    assert resp["status"]["code"] == 1001 and fingerprint.parse(resp) is None


def test_error_status_raises_before_caching(acr):
    a, _, post = acr
    post.body = {"status": {"msg": "Request count limit exceeded", "code": 3003, "version": "1.0"}}
    with pytest.raises(RuntimeError, match="3003"):
        a.identify(music())


def test_parse_flattens_documented_response():
    assert fingerprint.parse(DOC_OK) == {"artist": "Some Artist", "title": "Some Song", "score": 100,
                                         "play_offset_s": 9.04, "acr_id": "0123456789abcdef0123456789abcdef"}
    assert fingerprint.parse(NO_RESULT) is None


def test_requests_are_spaced_for_the_qps_limit(acr):
    a, cap, _ = acr
    t0 = time.monotonic()
    for _ in range(3):
        a.identify(music())
    assert cap["n"] == 3 and time.monotonic() - t0 >= 2 * (1 / fingerprint.ACR.QPS) - 0.05


def test_qps_exceeded_is_retried_once(acr):
    a, cap, post = acr
    post.bodies = [{"status": {"msg": "QpS limit exceeded", "code": 3015, "version": "1.0"}}, NO_RESULT]
    assert a.identify(music()) == NO_RESULT and cap["n"] == 2