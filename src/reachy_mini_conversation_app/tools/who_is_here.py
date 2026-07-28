"""Fork addition: recognize enrolled people by face. See FACES.md."""

import asyncio
import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies

logger = logging.getLogger(__name__)


class WhoIsHere(Tool):
    """Identify which known people are currently visible, by name."""

    name = "who_is_here"
    description = (
        "Look through the camera and identify BY NAME which of the people you know "
        "are currently visible. Use this when someone starts talking to you, when you "
        "hear a new voice, when someone greets you, when asked who is in the room, or "
        "before addressing someone so you can use their name. Returns the names of "
        "recognized people and how many unfamiliar faces are visible. The camera is "
        "live; each call reflects the current moment."
    )
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Run face recognition on the current camera frame."""
        if not deps.camera_enabled:
            logger.error("who_is_here: camera is disabled")
            return {"error": "Camera is disabled"}

        frame = deps.reachy_mini.media.get_frame()
        if frame is None:
            logger.error("who_is_here: no frame available")
            return {"error": "No frame available"}

        try:
            from reachy_mini_conversation_app.face_recognition import get_identifier

            identifier = await asyncio.to_thread(get_identifier)
            names, unknown = await asyncio.to_thread(identifier.identify, frame)
        except Exception as e:
            logger.exception("who_is_here failed")
            return {"error": f"face recognition failed: {e}"}

        if not identifier.names:
            return {
                "people": [],
                "unknown_faces": unknown,
                "note": "No one is enrolled yet - faces database is empty.",
            }

        logger.info("Tool call: who_is_here -> %s (+%d unknown)", names, unknown)
        return {"people": names, "unknown_faces": unknown}
