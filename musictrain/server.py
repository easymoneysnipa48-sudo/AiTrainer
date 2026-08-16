"""FastAPI REST backend + job queue (Advanced #41/#42).

Exposes the toolkit as an HTTP API so the dashboard (or any client) can trigger
long jobs (eval, generate, features) and poll their status — the CLI and UI
stay the primary interfaces, this is the automation surface.

* Jobs run on a background thread pool; ``POST /eval`` returns immediately with
  a ``job_id``, ``GET /jobs/{id}`` reports progress/result.
* The queue persists job records to ``metadata/jobs/*.json`` (gap #11) so a
  restart keeps job history; in-flight jobs are marked ``interrupted``.
* FastAPI + uvicorn are optional dependencies — the module imports lazily and
  prints install instructions if missing.

Run with:  ``musictrain serve --port 8000``
"""
from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional

from . import console
from .config import Config


# --------------------------------------------------------------------------- #
# Job queue (#42) with persistence (#11)
# --------------------------------------------------------------------------- #
class JobQueue:
    """Thread-based job queue with progress + result capture and disk persistence.

    ``root`` enables persistence (job records under ``<root>/metadata/jobs/``).
    ``None`` keeps the queue purely in memory (default, used in tests).
    """

    def __init__(self, root: Optional[Path] = None) -> None:
        self._jobs: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self._dir = Path(root) / "metadata" / "jobs" if root else None
        self._restore()

    # -- persistence ---------------------------------------------------- #
    def _path(self, job_id: str) -> Optional[Path]:
        return self._dir / f"{job_id}.json" if self._dir else None

    def _persist(self, job: dict) -> None:
        p = self._path(job["id"])
        if p is None:
            return
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(job, default=str))
        except OSError:  # noqa: BLE001 - persistence must never break the job
            pass

    def _restore(self) -> None:
        if self._dir is None or not self._dir.exists():
            return
        for p in sorted(self._dir.glob("*.json")):
            try:
                job = json.loads(p.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if job.get("status") in ("queued", "running", "cancelling"):
                job["status"] = "interrupted"
                job["error"] = "server restarted while job was in flight"
            self._jobs[job["id"]] = job

    # -- job lifecycle -------------------------------------------------- #
    def submit(self, fn: Callable, *args, **kwargs) -> str:
        job_id = uuid.uuid4().hex[:12]
        job = {"id": job_id, "status": "queued", "progress": 0.0,
               "message": "", "result": None, "error": None}
        with self._lock:
            self._jobs[job_id] = job
        self._persist(job)
        t = threading.Thread(target=self._run, args=(job_id, fn, args, kwargs), daemon=True)
        t.start()
        return job_id

    def _run(self, job_id: str, fn: Callable, args, kwargs) -> None:
        def progress(done: int, total: int) -> None:
            self.update(job_id, progress=done / max(total, 1), message=f"{done}/{total}")

        def cancel() -> bool:
            return self.status(job_id) in ("cancelling",)

        try:
            self.update(job_id, status="running", message="started")
            result = fn(*args, progress=progress, cancel=cancel, **kwargs)
            self.update(job_id, status="done", progress=1.0, result=result)
        except Exception as exc:  # noqa: BLE001
            self.update(job_id, status="failed", error=str(exc))

    def update(self, job_id: str, **fields) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(fields)
        self._persist(self._jobs[job_id])

    def status(self, job_id: str) -> Optional[str]:
        with self._lock:
            job = self._jobs.get(job_id)
            return job["status"] if job else None

    def get(self, job_id: str) -> Optional[dict]:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job["status"] == "running":
                job["status"] = "cancelling"
                self._persist(job)
                return True
            return False

    def list(self) -> List[dict]:
        with self._lock:
            return [
                {"id": j["id"], "status": j["status"], "progress": j["progress"],
                 "message": j["message"]}
                for j in sorted(self._jobs.values(), key=lambda x: x["id"])
            ]


QUEUE = JobQueue()


def _importable(module: str) -> bool:
    try:
        __import__(module)
        return True
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# App factory (#41)
# --------------------------------------------------------------------------- #
def create_app(cfg: Config, require_token: Optional[str] = None):
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import FileResponse, StreamingResponse
        from pydantic import BaseModel
    except ImportError:
        console.error("fastapi is not installed — run `uv pip install fastapi uvicorn` and retry.")
        raise

    app = FastAPI(title="musictrain API", version="1.0")
    queue = JobQueue(cfg.project_root)  # persistent job queue (#11)

    # -- optional bearer-token auth (#17) -------------------------------- #
    if require_token:
        @app.middleware("http")
        async def _auth(request: Request, call_next):
            auth = request.headers.get("Authorization", "")
            if auth != f"Bearer {require_token}":
                from fastapi.responses import JSONResponse
                return JSONResponse(status_code=401, content={"detail": "invalid token"})
            return await call_next(request)

    class EvalRequest(BaseModel):
        section: Optional[str] = None
        seeds: int = 1
        limit: int = 0

    class GenerateRequest(BaseModel):
        prompt: str
        seed: int = 0

    @app.get("/health")
    def health():
        return {"status": "ok", "root": str(cfg.project_root)}

    @app.get("/health/live")
    def health_live():
        return {"status": "alive", "jobs": len(queue._jobs)}

    @app.get("/ready")
    def ready():
        """Readiness: check the project layout + optional heavy deps are usable."""
        checks = {
            "project_root": (cfg.project_root / "metadata").exists(),
            "torch": _importable("torch"),
            "streamlit": _importable("streamlit"),
        }
        ready = checks["project_root"]
        status = "ready" if ready else "not_ready"
        return {"status": status, "checks": checks}

    @app.post("/generate/stream")
    def generate_stream(req: GenerateRequest):
        """NDJSON progress events while a clip is generated (progress surface)."""
        from .inference import generate

        def gen():
            yield b"{\"event\": \"start\"}\n"
            try:
                out = generate(cfg, req.prompt, seed=req.seed)
                yield (json.dumps({"event": "done", "path": out["path"],
                                  "duration": out.get("duration")}) + "\n").encode()
            except Exception as exc:  # noqa: BLE001
                yield (json.dumps({"event": "error", "message": str(exc)}) + "\n").encode()

        return StreamingResponse(gen(), media_type="application/x-ndjson")

    @app.post("/generate/stream-audio")
    def generate_stream_audio(req: GenerateRequest):
        """Real audio streaming (#10) — generates, then streams the WAV bytes in
        chunks so clients can begin buffering/playing before the response ends."""
        from .inference import generate

        def gen():
            out = generate(cfg, req.prompt, seed=req.seed)
            path = Path(out["path"])
            with path.open("rb") as fh:
                while True:
                    chunk = fh.read(64 * 1024)
                    if not chunk:
                        break
                    yield chunk

        return StreamingResponse(gen(), media_type="audio/wav",
                                 headers={"Content-Disposition": "attachment"})

    @app.get("/audio/{filename}")
    def audio(filename: str):
        """Serve a generated audio file (chunked by FileResponse with range support)."""
        base = cfg.project_root / "outputs"
        path = (base / filename).resolve()
        if not path.is_file() or base.resolve() not in path.parents:
            raise HTTPException(status_code=404, detail="audio not found")
        return FileResponse(path, media_type="audio/wav")

    @app.get("/inventory")
    def inventory():
        from .audio.inventory import inventory as inv

        return inv(cfg.project_root, cfg, which="clean", sha256=False)

    @app.get("/metrics")
    def metrics():
        from .metrics import compute

        return compute(cfg)

    @app.get("/leaderboard")
    def leaderboard():
        from .leaderboard import build

        return build(cfg)

    @app.post("/eval")
    def start_eval(req: EvalRequest):
        from .evalset import run_eval

        def run(progress=None, cancel=None):
            return run_eval(
                cfg, limit=req.limit, section=req.section, seeds=req.seeds,
                progress=progress, cancel=cancel,
            )

        return {"job_id": queue.submit(run)}

    @app.post("/generate")
    def start_generate(req: GenerateRequest):
        from .inference import generate

        def run(progress=None, cancel=None):
            return generate(cfg, req.prompt, seed=req.seed)

        return {"job_id": queue.submit(run)}

    @app.get("/jobs")
    def jobs():
        return {"jobs": queue.list()}

    @app.get("/jobs/{job_id}")
    def job(job_id: str):
        j = queue.get(job_id)
        if j is None:
            raise HTTPException(status_code=404, detail="job not found")
        return j

    @app.post("/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        if not queue.cancel(job_id):
            raise HTTPException(status_code=400, detail="job not running")
        return {"status": "cancelling"}

    @app.post("/cache/warm")
    def warm_cache():
        """Pre-pull the HF model weights (#13) so first generation is fast."""
        from .cache_warm import warm

        return warm(cfg)

    return app


def serve(cfg: Config, port: int = 8000, token: str = "") -> int:
    try:
        import uvicorn
    except ImportError:
        console.error("uvicorn is not installed — run `uv pip install uvicorn` and retry.")
        return 1
    app = create_app(cfg, require_token=token or None)
    console.info(f"Serving musictrain API at http://localhost:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port)
    return 0
