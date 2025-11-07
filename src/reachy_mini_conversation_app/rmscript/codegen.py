"""Code generator for ReachyMiniScript - creates executable Python functions."""

import time
from typing import List, Callable

from reachy_mini_conversation_app.rmscript.errors import Action, WaitAction


class CodeGenerator:
    """Generates executable Python functions from IR."""

    def generate(
        self, tool_name: str, description: str, ir: List[Action | WaitAction]
    ) -> Callable:
        """Generate executable function from IR."""

        def executable(mini: "ReachyMini") -> None:  # type: ignore # noqa: F821
            """Generate function to execute on robot."""
            for action in ir:
                if isinstance(action, WaitAction):
                    time.sleep(action.duration)

                elif isinstance(action, Action):
                    # Build parameters for goto_target
                    kwargs = {}

                    if action.head_pose is not None:
                        kwargs["head"] = action.head_pose

                    if action.antennas is not None:
                        kwargs["antennas"] = action.antennas

                    if action.body_yaw is not None:
                        kwargs["body_yaw"] = action.body_yaw

                    kwargs["duration"] = action.duration

                    if action.interpolation != "minjerk":
                        kwargs["method"] = action.interpolation

                    # Execute movement
                    if kwargs:  # Only call if we have parameters
                        mini.goto_target(**kwargs)

        # Set function metadata
        executable.__name__ = tool_name
        executable.__doc__ = description

        return executable
