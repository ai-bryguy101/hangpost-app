# Deploying the HuggingFace Space

The live demo at <https://huggingface.co/spaces/thisaiguybry/hangpost>
runs from the [`space/`](../space/) directory in this repo. The Space's
own git repo lives on HuggingFace — we push to it manually after editing
`space/` here, because there is no auto-deploy hook.

This doc is the single source of truth for:
- the operational facts about the Space (URL, owner, SHA pin)
- the recipe to deploy or redeploy from a fresh Codespace
- the two specific failure modes that have bitten us before, and how to
  diagnose them when they happen again

If you're an agent reading this, the headline you need is:

> **Don't trust HuggingFace's pip cache. After every push that changes
> `requirements.txt`, click Settings → Factory rebuild in the HF web UI.**

---

## The facts (don't re-derive these)

| | |
|---|---|
| HF Space URL | <https://huggingface.co/spaces/thisaiguybry/hangpost> |
| HF owner | `thisaiguybry` |
| HF Space name | `hangpost` (NOT `hangpost-matching-demo`) |
| Source files in this repo | [`space/`](../space/) |
| Package install pin | `space/requirements.txt` — pinned to a SHA on the public `main` branch |
| App entry point | `space/app.py` (Gradio Blocks UI) |
| Hardware | CPU basic (free tier) |
| SDK | Gradio (version pinned via the `sdk_version:` line in `space/README.md`) |

## The two failure modes that have bitten us

### 1. Vendored stale package in the Space's git repo (the import-resolution trap)

**Symptom.** The Space throws an `AttributeError` on a `MatchBreakdown`
field that exists in our code (e.g. `has_both_shared_background`), even
though the pinned SHA's `models.py` clearly has it.

**Root cause.** A previous deploy committed a *vendored* copy of the
`hangpost_matching/` Python package directly into the Space repo root.
Python's import resolution prefers a local package over a pip-installed
one, so `app.py` was importing from those stale vendored files instead
of the GitHub-pinned install.

**Fix.** Wipe everything except `.git` from the Space repo before copying
in our `space/` contents:

```bash
cd /tmp/hangpost-space
find . -mindepth 1 -maxdepth 1 ! -name '.git' -exec rm -rf {} +
cp -a /workspaces/hangpost-app/space/. .
```

The recipe below already does this. **Never** `cp -a` over an existing
Space without nuking the old contents first.

### 2. HuggingFace pip cache serving an old install (the cache trap)

**Symptom.** You push a new SHA in `requirements.txt`, but the Space
keeps behaving like the *previous* SHA's code is installed.

**Root cause.** HF caches the pip-install layer of the Space's Docker
image. A push that only changes the pinned SHA in `requirements.txt`
will produce a new layer, but if HF reuses the prior install layer
(intermittent — depends on cache invalidation rules) you'll get the
stale package.

**Fix.** After every push that changes `requirements.txt`, immediately:

1. Open <https://huggingface.co/spaces/thisaiguybry/hangpost>
2. Click **Settings** (top right)
3. Scroll to **Factory rebuild** → click → confirm

This rebuilds the image from scratch and busts the pip cache. Takes 2–4
minutes. **Skip this step at your own risk.**

---

## First-time deploy (fresh Codespace, ~10 minutes)

1. **Open a new Codespace** on branch `claude/cool-cori-X6EaH` (or
   whatever the active feature branch is) from the repo's `<> Code`
   button → Codespaces tab. Delete old Codespaces first if you're near
   your quota.

2. **Create a write-scoped HF token** at
   <https://huggingface.co/settings/tokens> if you don't already have
   one. Copy it — HF shows it once.

3. **Install + authenticate** in the Codespace terminal:

   ```bash
   pip install -U huggingface_hub   # the `[cli]` extra is deprecated; cli ships in the base package
   hf --version                     # confirm `hf` is on PATH
   git config --global credential.helper store
   hf auth login                    # paste the token; answer "y" to "Add token as git credential?"
   hf auth whoami                   # must print `thisaiguybry`
   ```

4. **Clone the empty Space repo** into a temp location:

   ```bash
   rm -rf /tmp/hangpost-space
   git clone https://huggingface.co/spaces/thisaiguybry/hangpost /tmp/hangpost-space
   ```

5. **Copy our `space/` contents over a clean slate**:

   ```bash
   cd /tmp/hangpost-space
   find . -mindepth 1 -maxdepth 1 ! -name '.git' -exec rm -rf {} +
   cp -a /workspaces/hangpost-app/space/. .
   ```

6. **Sanity-check before pushing** — these counts must match:

   ```bash
   wc -l app.py                                            # 180
   grep -c has_both_shared_background app.py               # 2
   grep -c _pick_showcase_profiles app.py                  # 0 (any phantom function name from prior deploys)
   grep '@' requirements.txt                               # SHA pin matches the package version that has the fields app.py needs
   ```

7. **Commit and push**:

   ```bash
   git add -A
   git status                       # review the diff before pushing
   git commit -m "Deploy from <branch-name>"
   git push                         # use HF token as password if prompted
   ```

8. **Factory rebuild** — go to the Space's Settings page in the browser
   and click **Factory rebuild**. (See failure mode 2 above for why.)

9. **Watch the build** at the Space URL. Expected sequence:
   - 🟡 Building (~2–4 min: pip-installs gradio + our package from GitHub)
   - 🟡 Application starting (~10s: gradio launch)
   - 🟢 Running (✅ demo is live; the dropdown should populate with profile names)

If the build errors, open the **Logs** tab on the Space page. Most
errors fall into the two failure modes above.

---

## Updating an already-deployed Space

Same recipe as "first-time deploy" steps 4–9, with one optional
shortcut: if the Space repo is still cloned in `/tmp/hangpost-space`
from a prior session, skip step 4.

If you changed package code (anything in `src/hangpost_matching/`), also
bump the SHA pin in `space/requirements.txt` before redeploying, then
factory-rebuild:

```bash
cd /workspaces/hangpost-app
git checkout main && git pull
NEW_SHA=$(git rev-parse HEAD)
sed -i "s/hangpost-app@[0-9a-f]\{40\}/hangpost-app@${NEW_SHA}/" space/requirements.txt
git add space/requirements.txt
git commit -m "Bump Space pin to ${NEW_SHA:0:7}"
git push origin <branch>
# Then re-run steps 4–9 above.
```

The SHA pin only needs bumping when the package's public surface
(anything `space/app.py` imports) changes. README-only or test-only
changes don't require a re-pin.

---

## Why this lives in its own doc

The CLAUDE.md persistent-context file points here for two reasons:

1. The recipe is verbose enough that inlining it in CLAUDE.md would
   crowd out the modeling context that matters more often. Most
   sessions don't touch deployment.

2. The failure modes above are *operational* knowledge — they aren't
   in the code itself, they're in how the code interacts with the HF
   platform. Future-me reading this in six months will need the
   "vendored package trap" story written down because it's not
   obvious from `space/app.py` alone.
