"""Regression test for the dashboard job runners (cancel button + progress).

The dashboard module calls ``main()`` at import time, so these tests load the
module source with that trailing call stripped and then exercise the real
``_run_job_cancellable`` / ``_run_live`` functions under Streamlit's AppTest.

The key regression: the Cancel button was rendered *inside* the polling loop
with a fixed ``key``, so each tick re-registered the same widget key and raised
``StreamlitDuplicateElementKey``. It must now be rendered once, outside the loop.
"""
from __future__ import annotations

import re
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]

_LOADER = """
import sys, time, re
from pathlib import Path
sys.path.insert(0, {root!r})

src = (Path({root!r}) / "musictrain" / "dashboard.py").read_text()
# strip the module-level main() call so we can import the helpers cleanly
src = re.sub(r"\\n\\nmain\\(\\)\\s*$", "\\n", src)
ns = {{"__file__": str(Path({root!r}) / "musictrain" / "dashboard.py"),
      "__name__": "dashboard_under_test", "__package__": "musictrain"}}
exec(compile(src, ns["__file__"], "exec"), ns)

def _slow_worker():
    time.sleep(0.4)
    return "ok"

result = ns["_run_job_cancellable"]("test job", _slow_worker)
assert result == "ok", f"unexpected result: {{result!r}}"
print("JOB_OK")
""".format(root=str(ROOT))


def test_run_job_cancellable_no_duplicate_key():
    at = AppTest.from_string(_LOADER).run()
    assert not at.exception, f"job runner raised: {[str(e.value) for e in at.exception]}"
