"""Exercise the public service contract against a running stack, with synthetic QA images."""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path
from uuid import uuid4

import httpx
from PIL import Image, ImageDraw


def specimen(color: str) -> bytes:
    image = Image.new("RGB", (320, 220), "#e8eaed")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((30, 60, 290, 170), 20, fill=color)
    draw.rectangle((80, 35, 220, 95), fill=color)
    draw.ellipse((60, 150, 100, 190), fill="#20242b")
    draw.ellipse((220, 150, 260, 190), fill="#20242b")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def expect(response: httpx.Response, statuses: tuple[int, ...] = (200, 201, 202)) -> dict:
    if response.status_code not in statuses:
        raise RuntimeError(
            f"{response.request.method} {response.request.url.path}: "
            f"HTTP {response.status_code}: {response.text[:500]}"
        )
    return response.json()


def await_job(client: httpx.Client, job: dict, timeout: int = 90) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = expect(client.get(f"/api/v1/jobs/{job['id']}"))
        if current["status"] == "completed":
            return current["result"]
        if current["status"] == "failed":
            raise RuntimeError(f"Job {current['kind']} failed: {current.get('error')}")
        time.sleep(0.25)
    raise TimeoutError(f"Job {job['id']} did not complete within {timeout}s")


def run(url: str, token: str) -> dict:
    with httpx.Client(base_url=url, headers={"Authorization": f"Bearer {token}"}, timeout=90) as client:
        capabilities = expect(client.get("/api/v1/capabilities"))
        gallery = expect(
            client.post(
                "/api/v1/galleries",
                json={
                    "name": f"QA · synthetic shapes · {uuid4().hex[:8]}",
                    "description": "Synthetic service smoke test; not a ReID benchmark or real vehicle dataset.",
                },
            )
        )
        blue, red = specimen("#597bb0"), specimen("#bb625c")
        imported = await_job(
            client,
            expect(
                client.post(
                    f"/api/v1/galleries/{gallery['id']}/imports",
                    files=[
                        ("files", ("blue.png", blue, "image/png")),
                        ("files", ("red.png", red, "image/png")),
                    ],
                )
            ),
        )
        expect(
            client.post(
                f"/api/v1/galleries/{gallery['id']}/versions/{imported['version_id']}/activate",
                json={"expected_active_version_id": None},
            )
        )
        key = str(uuid4())
        query = {"gallery_id": gallery["id"], "bbox": "[20,25,280,180]", "top_n": "10"}

        def search() -> dict:
            return expect(
                client.post(
                    "/api/v1/searches",
                    data=query,
                    files={"image": ("query.png", blue, "image/png")},
                    headers={"Idempotency-Key": key},
                )
            )

        result = search()
        assert result["decision"] == "calibration_unavailable", "Stub must never identify a vehicle"
        assert result["accepted_candidates"] == []
        assert len(result["ranked_candidates"]) == 2
        assert search()["id"] == result["id"], "Retry must reuse the persisted operation"
        candidate = result["ranked_candidates"][0]
        expect(
            client.post(
                f"/api/v1/searches/{result['id']}/reviews",
                json={
                    "observation_id": candidate["observation_id"],
                    "verdict": "uncertain",
                    "note": "Проверка сервиса: синтетическое изображение, модель-заглушка.",
                },
            )
        )
        explanation = await_job(
            client,
            expect(
                client.post(
                    f"/api/v1/searches/{result['id']}/explanations",
                    json={"observation_id": candidate["observation_id"]},
                )
            ),
        )
        exported = {}
        for format_name in ("json", "csv", "pdf"):
            artifact = await_job(
                client,
                expect(
                    client.post(
                        f"/api/v1/searches/{result['id']}/exports",
                        json={"format": format_name},
                    )
                ),
            )
            download = client.get(artifact["download_url"])
            download.raise_for_status()
            if format_name == "pdf":
                assert download.content.startswith(b"%PDF-")
            elif format_name == "json":
                assert download.json()["search"]["id"] == result["id"]
            else:
                assert b"observation_id" in download.content
            exported[format_name] = len(download.content)
        replacement = await_job(
            client,
            expect(
                client.post(
                    f"/api/v1/galleries/{gallery['id']}/imports",
                    files=[("files", ("green.png", specimen("#60877e"), "image/png"))],
                )
            ),
        )
        expect(
            client.post(
                f"/api/v1/galleries/{gallery['id']}/versions/{replacement['version_id']}/activate",
                json={"expected_active_version_id": imported["version_id"]},
            )
        )
        assert search()["gallery_version_id"] == imported["version_id"]
        replay = expect(client.post(f"/api/v1/searches/{result['id']}/replays"))
        assert replay["gallery_version_id"] == result["gallery_version_id"]
        assert [x["observation_id"] for x in replay["ranked_candidates"]] == [
            x["observation_id"] for x in result["ranked_candidates"]
        ]
        assert len(expect(client.get(f"/api/v1/searches/{result['id']}"))["reviews"]) == 1
        assert client.get(candidate["image_url"]).status_code == 200
        with httpx.Client(base_url=url, timeout=10) as anonymous:
            assert anonymous.get(candidate["image_url"]).status_code == 401
        return {
            "status": "passed",
            "gallery_id": gallery["id"],
            "search_id": result["id"],
            "model_id": result["model_id"],
            "decision": result["decision"],
            "export_bytes": exported,
            "explanation_method": explanation.get("method"),
            "capabilities_available": bool(capabilities),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8090")
    parser.add_argument(
        "--token-file", type=Path, default=Path(__file__).resolve().parents[1] / ".local/access-token.txt"
    )
    args = parser.parse_args()
    print(json.dumps(run(args.url, args.token_file.read_text().strip()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
