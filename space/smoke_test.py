"""Pre-deploy smoke test for the HuggingFace Space.

CI never installs gradio (it tests the `hangpost_matching` package only), so
nothing in the normal test suite validates that `app.py` still works against
the gradio version pinned in `requirements.txt`. Run this before pushing the
Space — especially after a gradio major bump (e.g. the Dependabot 5 -> 6 PR).

What it checks, cheapest first:
  1. gradio imports and is the major version you expect.
  2. `app.py` imports cleanly — this runs the CSV load AND builds the whole
     `gr.Blocks` UI, so any changed component signature (Dropdown, Blocks,
     Row, .change/.load) blows up here.
  3. The `rank_for_source` callback returns sensible markdown for a real
     profile and for the empty selection.
  4. (Optional, --launch) actually boots the server and fires demo.load(),
     catching launch-time / schema-generation breakage, then shuts down.

Usage:
    python space/smoke_test.py            # steps 1-3 (no server)
    python space/smoke_test.py --launch   # also boot + close the server
"""

from __future__ import annotations

import sys
from pathlib import Path

SPACE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SPACE_DIR))


def main(do_launch: bool) -> int:
    # 1. gradio present + version.
    try:
        import gradio as gr
    except ImportError:
        print("FAIL: gradio is not installed. `pip install -r space/requirements.txt`")
        return 1
    print(f"gradio version: {gr.__version__}")
    major = gr.__version__.split(".")[0]
    if major != "6":
        print(f"WARNING: expected gradio 6.x, got {gr.__version__}")

    # 2. Importing the app builds the whole UI (construction-time API check).
    import app

    assert isinstance(app.demo, gr.Blocks), "app.demo is not a gr.Blocks"
    assert app.DROPDOWN_CHOICES, "no dropdown choices were built from the CSV"
    print(f"app imported OK: {len(app.PROFILES)} profiles, "
          f"{len(app.DROPDOWN_CHOICES)} dropdown choices")

    # 3. Exercise the callback directly (no server needed).
    first_id = app.DROPDOWN_CHOICES[0][1]
    out = app.rank_for_source(first_id)
    assert isinstance(out, str) and out.strip(), "callback returned empty output"
    assert "Tier" in out, "callback output is missing the tier badges"
    empty = app.rank_for_source("")
    assert "Pick a source profile" in empty, "empty-selection path changed"
    print(f"rank_for_source OK: {len(out)} chars for {first_id!r}")

    # 4. Optional: boot the real server, which runs demo.load(), then close.
    if do_launch:
        app.demo.launch(prevent_thread_lock=True, show_error=True)
        try:
            print("demo.launch() OK (server started)")
        finally:
            app.demo.close()
            print("demo.close() OK")

    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main(do_launch="--launch" in sys.argv[1:]))
