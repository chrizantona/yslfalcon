"""Import a complete synthetic 750-image gallery through the public chunked API."""

import argparse
import json
from pathlib import Path
from uuid import uuid4

import httpx
from smoke import await_job, expect, specimen


def run(url: str, token: str, count: int = 750) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    pixels = [specimen(f"#{value:02x}6699") for value in range(16, 240, 16)]
    with httpx.Client(base_url=url, headers=headers, timeout=120) as client:
        gallery = expect(
            client.post(
                "/api/v1/galleries",
                json={"name": f"QA · full synthetic gallery · {uuid4().hex[:8]}"},
            )
        )
        base = f"/api/v1/galleries/{gallery['id']}"
        manifest = "image_id,x,y,w,h\n" + "".join(f"sample-{i},20,25,280,180\n" for i in range(count))
        key = str(uuid4())
        session = expect(
            client.post(
                f"{base}/import-sessions", json={"manifest_csv": manifest}, headers={"Idempotency-Key": key}
            )
        )
        repeat = expect(
            client.post(
                f"{base}/import-sessions", json={"manifest_csv": manifest}, headers={"Idempotency-Key": key}
            )
        )
        assert repeat["id"] == session["id"]
        endpoint = f"{base}/import-sessions/{session['id']}"
        assert client.post(f"{endpoint}/finalize").status_code == 409
        chunk_size = min(100, session["limits"]["max_chunk_files"])
        for start in range(0, count, chunk_size):
            batch = [
                ("files", (f"sample-{i}.png", pixels[i % len(pixels)], "image/png"))
                for i in range(start, min(count, start + chunk_size))
            ]
            progress = expect(client.post(f"{endpoint}/files", files=batch))
            assert progress["uploaded_count"] == min(count, start + chunk_size)
            retried = expect(client.post(f"{endpoint}/files", files=batch))
            assert retried["uploaded_count"] == progress["uploaded_count"]
        # The persisted session is sufficient for another connection to resume/finalize.
    with httpx.Client(base_url=url, headers=headers, timeout=120) as client:
        resumed = expect(client.get(endpoint))
        assert resumed["validation"]["ready"] and resumed["missing_image_ids"] == []
        job = expect(client.post(f"{endpoint}/finalize"))
        assert expect(client.post(f"{endpoint}/finalize"))["id"] == job["id"]
        imported = await_job(client, job, timeout=600)
        versions = expect(client.get(f"{base}/versions"))["items"]
        assert len(versions) == 1 and versions[0]["count"] == count
        assert expect(client.get(base))["active_version_id"] is None
        expect(
            client.post(
                f"{base}/versions/{imported['version_id']}/activate",
                json={"expected_active_version_id": None},
            )
        )
        result = expect(
            client.post(
                "/api/v1/searches",
                data={"gallery_id": gallery["id"], "bbox": "[20,25,280,180]", "top_n": "10"},
                files={"image": ("query.png", pixels[0], "image/png")},
            )
        )
        assert result["decision"] == "calibration_unavailable" and result["accepted_candidates"] == []
        assert len(result["ranked_candidates"]) == 10
        return {
            "status": "passed",
            "synthetic_observations": count,
            "gallery_id": gallery["id"],
            "import_session_id": session["id"],
            "version_id": imported["version_id"],
            "search_id": result["id"],
            "checks": [
                "chunk_retry",
                "resume",
                "missing_files_block",
                "single_finalize",
                "explicit_activation",
                "search",
            ],
            "model_quality_verified": False,
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
