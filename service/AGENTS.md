# Service development

This directory implements the operator service. The research code outside it remains independent.

## Current scope

- The trained model is intentionally disconnected at the user's request. Use the explicit stub/preview adapter only. Do not load weights, add model SDKs or run experiments.
- All visible data comes from actual service operations. Never seed invented cameras, throughput, accuracy, accepted identities or sample search histories into the UI.
- The stub is visibly labelled. It cannot produce a calibrated vehicle identification or a claim of real model explainability.

## Design and code

- Keep domain operations behind small typed interfaces. HTTP, persistence and the browser must not each implement their own ranking or acceptance rules.
- Separate immutable machine results from operator reviews. Pin snapshot/model/policy on the first request and keep it on retry/replay.
- Every object route is scoped to its workspace. Cookie mutations validate the origin and CSRF header; integrations use bearer authentication.
- Use explicit structured errors; never turn infrastructure failure into a model refusal.
- Prefer vertical user scenarios and tests through public interfaces over private-method mocks and generic frameworks.
- TypeScript strict mode; avoid `any`, swallowed errors, duplicated domain types and decorative abstractions. React components express user tasks; API transport stays in one module.
- Interface follows a restrained desktop visual system: local Unbounded headings, Manrope body text and IBM Plex Mono data, clear typography, plain near-black (#0d0d0e) work surfaces without section dividers, translucent white dots behind search results with Gaussian cursor response and readable captions, white accent text actions with the neutral palette from st-falcon.ru and compact text-only side navigation without icons, with accent-colored hover text, flat upload prompts with white vector artwork, restrained corner radii, consistent spacing, keyboard focus and reduced-motion support.
- Hover typing uses one shared controller: retain original DOM text, layout and accessible names; never type into editable values. Restore labels before activation and when motion is disabled.
- Keep all runtime assets local. No CDN fonts, cloud model downloads or telemetry services.

## Checks

Run commands from `service/`:

```sh
.venv/bin/python -m pytest
.venv/bin/ruff check backend tests scripts
.venv/bin/ruff format --check backend tests scripts
npm --prefix frontend run build
```

Check the actual Compose stack and public API workflow before declaring the service runnable. Document unavailable checks honestly. Secrets, local volumes, model files, databases and generated test artifacts are excluded from Git.
