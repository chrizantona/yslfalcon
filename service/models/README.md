# GPU runtime bundle handoff

The default service never reads this directory and keeps using the explicit preview adapter.
For the optional GPU profile, mount one external bundle read-only and provide its separately built
runtime image. The bundle is owned by the model team and is not committed to Git or copied into the
ordinary service image.

Copy `runtime-bundle.example.json` to `runtime-bundle.json` inside the external bundle. List every
regular file in that directory with its exact byte size, SHA-256 and role. Files with role `weight`
are summed against the conservative 2,000,000,000-byte service gate; auxiliary and reranker weights
must also use this role. Unlisted files, symlinks, path traversal, mutable `latest` images and hash
mismatches are rejected before Compose starts.

See `docs/GPU_RUNTIME.md` for the HTTP contract and launch procedure.
