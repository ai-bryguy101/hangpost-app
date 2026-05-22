# Security Policy

## Scope

The Hangpost Matching Engine is a research / portfolio project: a
prototype ranker, an offline evaluation harness, and an optional
FastAPI service. It has **not** been hardened for production use and
should not be deployed as-is to handle real user traffic.

That said, we take supply-chain and code-execution risks seriously
because anyone who installs this package or runs the Docker image is
trusting whatever this repository contains.

## Reporting a vulnerability

If you find a vulnerability — credential leakage in committed
history, a remote code execution path through the FastAPI server, a
prompt-injection vector through the LLM-judge pipeline, an
arbitrary-file-read via `scripts/label.py`, etc. — please report it
**privately**:

- Open a GitHub Security Advisory at
  <https://github.com/ai-bryguy101/hangpost-app/security/advisories/new>,
  **or**
- Email the repository owner directly (see the GitHub profile at
  <https://github.com/ai-bryguy101>).

Do **not** open a public issue or pull request — that would expose
the vulnerability to anyone watching the repo before a fix is ready.

## Response expectations

Because this is a portfolio repository maintained on a best-effort
basis:

- **Acknowledgement** within 7 days of report.
- **Triage and fix timeline** depends on severity — critical issues
  (RCE, credential exfiltration, secret commit) get priority over
  hardening suggestions.
- Coordinated disclosure after a fix is published; reporters are
  credited in the changelog unless they prefer anonymity.

## Known caveats (not vulnerabilities, but worth flagging)

These are documented design choices, not security bugs:

- **The FastAPI server has no authentication, no rate limiting, and
  no CORS configuration by default.** It is intended as a starting
  point for production deployment, not a production deployment itself.
  Anyone fronting it on the public internet must add at least:
  - An auth layer (API key header, JWT, or upstream proxy).
  - A rate limit (per-IP and per-key).
  - A CORS allowlist if a browser frontend will call `/rank`.
  - TLS termination (the included Dockerfile ships HTTP on port 8000).
- **The `/rank` endpoint caps `candidates` at `HANGPOST_MAX_CANDIDATES`
  (default 1000)** to prevent a single oversized payload from pinning
  a worker. Tune this for your deployment.
- **The LLM-judge pipeline sends profile data to the Anthropic API.**
  Profile contents in `data/test_profiles.csv` are synthetic and safe
  to send. If you point `scripts/label.py` at real user data, you
  inherit your own data-handling obligations (DPA, retention, etc.).
- **`models/learned_ranker.joblib` is gitignored and regenerable** —
  do not pickle untrusted model files. `joblib.load()` executes
  arbitrary code from pickle, so only load models you trained
  yourself or trust the provenance of.

## Supported versions

| Version | Supported? |
|---------|-----------|
| 0.1.x   | ✅ Yes      |
| < 0.1.0 | ❌ No       |

The project follows [semver](https://semver.org/) — patch and minor
releases on the current major track get security fixes; older majors
get them only by re-publishing forward.
