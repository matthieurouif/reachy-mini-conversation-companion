"""Fork addition: live "what does Reachy see" web page with face names.

Serves http://<robot>:8001 with the camera stream, a box around each face,
and the recognized person's name. Recognition only runs while someone is
watching the page, so it costs nothing otherwise.

Configure with REACHY_FACE_VIEW_PORT (default 8001, "off" to disable).
"""

from __future__ import annotations

import logging
import os
import threading
import time

import cv2

logger = logging.getLogger(__name__)

_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Reachy sees</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 body{margin:0;background:#111;color:#eee;font-family:system-ui,sans-serif;text-align:center}
 h1{font-size:1.1rem;font-weight:600;margin:.8rem 0 .4rem}
 img{max-width:100%;border-radius:8px}
 #people{margin:.6rem;font-size:1.05rem;color:#7fd77f;min-height:1.4em}
</style></head>
<body>
<h1>&#129302; What Reachy sees</h1>
<div id="people">&hellip;</div>
<img src="/stream" alt="camera stream">
<script>
 setInterval(async()=>{try{
   const r=await fetch('/people');const d=await r.json();
   const el=document.getElementById('people');
   const parts=[];
   if(d.people.length)parts.push('Recognized: <b>'+d.people.join(', ')+'</b>');
   if(d.unknown)parts.push(d.unknown+' unknown face'+(d.unknown>1?'s':''));
   el.innerHTML=parts.length?parts.join(' &middot; '):'No face in view';
 }catch(e){}},1000);
</script>
</body></html>"""


def maybe_start(robot) -> None:
    """Start the viewer server thread unless disabled by env."""
    port_s = os.getenv("REACHY_FACE_VIEW_PORT", "8001").strip().lower()
    if port_s in ("", "0", "off", "false", "no"):
        logger.info("Face view disabled (REACHY_FACE_VIEW_PORT=%s)", port_s)
        return
    port = int(port_s)
    threading.Thread(
        target=_serve, args=(robot, port), daemon=True, name="face-view"
    ).start()


def _serve(robot, port: int) -> None:
    try:
        import uvicorn
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

        from reachy_mini_conversation_app.face_recognition import get_identifier

        app = FastAPI()
        state = {"people": [], "unknown": 0, "ts": 0.0}

        def frames():
            identifier = get_identifier()
            while True:
                frame = robot.media.get_frame()
                if frame is None:
                    time.sleep(0.3)
                    continue
                frame = frame.copy()
                faces = identifier.identify_detailed(frame)
                state["people"] = [f["name"] for f in faces if f["name"]]
                state["unknown"] = sum(1 for f in faces if not f["name"])
                state["ts"] = time.time()
                for f in faces:
                    x, y, w, h = f["box"]
                    color = (0, 200, 0) if f["name"] else (0, 140, 255)
                    label = f["name"] or "?"
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(
                        frame, label, (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                    )
                ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    yield (
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                        + jpg.tobytes()
                        + b"\r\n"
                    )
                time.sleep(0.4)  # ~2 fps keeps the Pi comfortable

        @app.get("/")
        def index() -> HTMLResponse:
            return HTMLResponse(_PAGE)

        @app.get("/stream")
        def stream() -> StreamingResponse:
            return StreamingResponse(
                frames(), media_type="multipart/x-mixed-replace; boundary=frame"
            )

        @app.get("/people")
        def people() -> JSONResponse:
            return JSONResponse(state)

        logger.info("Face view available at http://0.0.0.0:%d", port)
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
    except Exception:
        logger.exception("Face view server failed")
