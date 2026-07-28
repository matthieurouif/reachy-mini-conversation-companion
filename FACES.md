# Face recognition (fork addition)

*This fork of [pollen-robotics/reachy_mini_conversation_app](https://github.com/pollen-robotics/reachy_mini_conversation_app)
adds a `who_is_here` tool: Reachy identifies the people in the room **by
name** with fully local face recognition, greets them personally, and uses
the built-in memory to store facts about each person.*

## How it works

- [face_recognition.py](src/reachy_mini_conversation_app/face_recognition.py)
  runs OpenCV YuNet (detection) + SFace (embeddings) locally — no cloud for
  vision. Enrolled people live in `data/faces.npz` (gitignored).
- [tools/who_is_here.py](src/reachy_mini_conversation_app/tools/who_is_here.py)
  exposes it to the conversation AI: when someone speaks, the model calls the
  tool, gets `{"people": ["Alice"], "unknown_faces": 0}`, and greets Alice by
  name.
- [profiles/companion](profiles/companion/) is a personality wired for this:
  identify whoever talks to you, greet by name, remember facts per person
  (using the app's own `remember`/`forget` long-term memory).

## Setup

```bash
pip install -e ".[faces]"    # on top of the app's normal install
```

Enroll your people from photos (3+ photos each, their face biggest in frame):

```bash
python scripts/enroll_faces.py --batch ~/Documents/people/
# people/Alice/*.jpg, people/Bob/*.jpg, ... (subfolder name = person's name)
```

Already enrolled in the [reachy-mini-companion](https://github.com/matthieurouif/reachy-mini-companion)
repo? The database format is identical — copy its `data/faces.npz` into
`data/` here, or point both repos at one shared folder:

```bash
export REACHY_DATA_DIR=/path/to/reachy-mini-companion/data
```

## Run with the companion profile

```bash
REACHY_MINI_CUSTOM_PROFILE=companion <however you normally launch the app>
```

The ONNX models auto-download on first use. Tune matching strictness with
`FACE_MATCH_THRESHOLD` (default 0.40; higher = stricter).

## Keeping up with upstream

Fork changes are isolated: `face_recognition.py`, `tools/who_is_here.py`,
`profiles/companion/`, `scripts/enroll_faces.py`, this file, plus marked
blocks in `pyproject.toml` / `.gitignore` / `README.md`.

```bash
git fetch upstream
git merge upstream/main
```
