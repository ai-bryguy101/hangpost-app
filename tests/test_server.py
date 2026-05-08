"""Smoke tests for the FastAPI deployment.

These tests are skipped when the `[serve]` extra is not installed (CI's
default `[dev]` install). When fastapi is present they spin up the app
in rules-only mode via `TestClient` and exercise the request → response
shape.
"""

import os

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="module")
def client() -> TestClient:
    os.environ["HANGPOST_MODE"] = "rules"
    from hangpost_matching.server import app

    with TestClient(app) as test_client:
        yield test_client


def test_healthz_returns_ok(client: TestClient) -> None:
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_describes_active_mode(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "hangpost-matching"
    assert body["mode"] == "rules"


def test_rank_orders_strong_match_above_weak(client: TestClient) -> None:
    payload = {
        "source": {
            "user_id": "u0",
            "interests": ["hiking", "coding"],
            "liked_topics": ["tech", "travel"],
            "location": "denver",
            "age": 28,
            "mutual_friend_ids": ["a", "b", "c"],
        },
        "candidates": [
            {
                "user_id": "weak",
                "interests": ["chess"],
                "liked_topics": ["finance"],
                "location": "seattle",
                "age": 50,
            },
            {
                "user_id": "strong",
                "interests": ["hiking", "coding"],
                "liked_topics": ["tech", "travel"],
                "location": "denver",
                "age": 28,
                "mutual_friend_ids": ["a", "b"],
            },
        ],
    }

    response = client.post("/rank", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "rules"
    assert [item["user_id"] for item in body["results"]] == ["strong", "weak"]
    # The strong match should also have its mutual-friend signal exposed.
    strong_item = body["results"][0]
    assert strong_item["has_mutual_friends"] is True
    assert strong_item["total_score"] > body["results"][1]["total_score"]


def test_rank_with_empty_candidate_list_returns_empty_results(
    client: TestClient,
) -> None:
    payload = {
        "source": {"user_id": "u0", "age": 30},
        "candidates": [],
    }

    response = client.post("/rank", json=payload)

    assert response.status_code == 200
    assert response.json() == {"mode": "rules", "results": []}
