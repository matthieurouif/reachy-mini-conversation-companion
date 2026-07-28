"""Enroll people from photos so the who_is_here tool can recognize them.

One person from a folder of their photos:
    python scripts/enroll_faces.py --name Alice --photos path/to/alice_photos/

Everyone at once (subfolder name = person's name):
    python scripts/enroll_faces.py --batch path/to/people/
    # people/Alice/*.jpg, people/Bob/*.jpg, ...

Needs 3+ photos per person where their face is the biggest in the image.
Embeddings are stored in data/faces.npz (gitignored — faces stay local).
The same file is used by the reachy-mini-companion repo; you can also copy
its faces.npz here, or point both repos at one place with REACHY_DATA_DIR.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reachy_mini_conversation_app.face_recognition import (  # noqa: E402
    DATA_DIR,
    FACES_DB,
    FaceIdentifier,
)


def load_db():
    """Return (names, embeddings) from data/faces.npz, empty if missing."""
    if FACES_DB.exists():
        db = np.load(FACES_DB)
        return [str(n) for n in db["names"]], db["embeddings"].astype(np.float32)
    return [], np.zeros((0, 128), dtype=np.float32)


def save_person(name: str, embedding: np.ndarray) -> None:
    """Add or replace one person's embedding in the database."""
    names, embeddings = load_db()
    if name in names:
        embeddings[names.index(name)] = embedding
    else:
        names.append(name)
        embeddings = np.vstack([embeddings, embedding[None, :]])
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(FACES_DB, names=np.array(names), embeddings=embeddings)
    print(f"Saved. Known people ({len(names)}): {', '.join(names)}")


def embed_folder(identifier: FaceIdentifier, folder: Path) -> list:
    """Extract one embedding per usable photo (biggest face wins)."""
    feats = []
    for img_path in sorted(folder.iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
            continue
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        faces = identifier.detect_faces(frame)
        if len(faces) == 0:
            print(f"  no face in {img_path.name}, skipping")
            continue
        det = max(faces, key=lambda f: f[2] * f[3])
        feats.append(identifier.embed(frame, det))
        print(f"  captured from {img_path.name}")
    return feats


def average(feats: list) -> np.ndarray:
    """Average embeddings into one normalized vector."""
    mean = np.mean(feats, axis=0)
    mean /= np.linalg.norm(mean) + 1e-9
    return mean.astype(np.float32)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", help="Person's first name")
    parser.add_argument("--photos", type=Path, help="Folder of photos of this person")
    parser.add_argument(
        "--batch", type=Path, help="Folder of per-person subfolders (subfolder = name)"
    )
    args = parser.parse_args()

    identifier = FaceIdentifier()

    if args.batch:
        folders = sorted(p for p in args.batch.iterdir() if p.is_dir())
        if not folders:
            sys.exit(f"No subfolders in {args.batch}. Expected one folder per person.")
        skipped = []
        for folder in folders:
            print(f"\n=== {folder.name} ===")
            feats = embed_folder(identifier, folder)
            if len(feats) < 3:
                skipped.append(f"{folder.name} ({len(feats)} usable photos, need 3+)")
                continue
            save_person(folder.name, average(feats))
        if skipped:
            print("\nSKIPPED - add more/clearer photos and re-run:")
            for s in skipped:
                print(f"  - {s}")
        return

    if not (args.name and args.photos):
        parser.error("use --batch DIR, or --name NAME --photos DIR")
    feats = embed_folder(identifier, args.photos)
    if len(feats) < 3:
        sys.exit(f"Only {len(feats)} usable photos - need at least 3.")
    save_person(args.name, average(feats))


if __name__ == "__main__":
    main()
