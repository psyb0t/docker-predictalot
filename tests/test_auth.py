from __future__ import annotations

from fastapi.testclient import TestClient


# Auth is enforced uniformly across every forecast endpoint; we cover the
# univariate route as a representative — same Depends(check_bearer) wires
# every other type router.
URL = "/v1/univariate/forecast"
PAYLOAD = {"model": "chronos-2", "context": [[1.0, 2.0]], "config": {"horizon": 1}}


class TestBearerAuth:
    def test_no_token_rejected(self, client: TestClient) -> None:
        resp = client.post(URL, json=PAYLOAD)
        assert resp.status_code == 401

    def test_wrong_token_rejected(self, client: TestClient) -> None:
        resp = client.post(URL, headers={"Authorization": "Bearer wrong"}, json=PAYLOAD)
        assert resp.status_code == 401

    def test_correct_token_accepted(self, client: TestClient) -> None:
        resp = client.post(URL, headers={"Authorization": "Bearer testtoken"}, json=PAYLOAD)
        assert resp.status_code == 200

    def test_query_param_token_accepted(self, client: TestClient) -> None:
        resp = client.post(f"{URL}?apiToken=testtoken", json=PAYLOAD)
        assert resp.status_code == 200

    def test_open_mode_allows_unauthed(self, open_client: TestClient) -> None:
        resp = open_client.post(URL, json=PAYLOAD)
        assert resp.status_code == 200

    def test_non_ascii_token_rejected_not_crashed(self, client: TestClient) -> None:
        """hmac.compare_digest on str raises TypeError for non-ASCII; the
        auth path must encode to bytes so a crafted bad token returns 401
        instead of crashing into a 500.

        Sent via percent-encoded ``apiToken`` query so the bytes are valid
        ASCII on the wire and decode to non-ASCII server-side. This is the
        crash vector — any unauthenticated client can trigger it.
        """
        # caf%C3%A9 decodes to "café"
        resp = client.post(f"{URL}?apiToken=caf%C3%A9-bogus", json=PAYLOAD)
        assert resp.status_code == 401
