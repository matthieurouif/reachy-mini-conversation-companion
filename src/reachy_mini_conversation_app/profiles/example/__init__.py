"""Example of a profile built using a custom tool and custom instructions."""

import logging
from typing import Any, Dict

from reachy_mini_conversation_app.rmscript import create_tool_from_rmscript


logger = logging.getLogger(__name__)


create_tool_from_rmscript("antennas_dance.rmscript")
create_tool_from_rmscript("check_behind.rmscript")
create_tool_from_rmscript("greet_with_sound.rmscript")



