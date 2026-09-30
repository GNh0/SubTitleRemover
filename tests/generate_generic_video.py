"""Generate a neutral video with a scene sign and a separate burned-in caption."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tempfile

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FFMPEG = ROOT / "third_party" / "ffmpeg9" / "ffmpeg-9.0.1-essentials_build" / "bin" / "ffmpeg.exe"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve(strict=False)
    if output.exists():
        parser.error("Output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "work").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="generic-sample-", dir=ROOT / "work") as folder:
        silent = Path(folder) / "video.mp4"
        writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"),
                                 15, (640, 360))
        if not writer.isOpened():
            raise RuntimeError("Could not create sample video")
        x = np.linspace(0, 1, 640)[None, :]
        y = np.linspace(0, 1, 360)[:, None]
        for index in range(45):
            frame = np.empty((360, 640, 3), dtype=np.uint8)
            frame[:, :, 0] = (45 + x * 55 + index * 0.2).astype(np.uint8)
            frame[:, :, 1] = (90 + y * 65 + index * 0.3).astype(np.uint8)
            frame[:, :, 2] = (120 + x * 30 + y * 20).astype(np.uint8)
            cv2.circle(frame, (420 + index // 3, 155), 34, (70, 160, 220), -1)
            cv2.putText(frame, "SCENE SIGN", (24, 280), cv2.FONT_HERSHEY_SIMPLEX,
                        .72, (20, 220, 245), 2, cv2.LINE_AA)
            if 10 <= index < 38:
                caption = "A sample subtitle appears here"
                text_width = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX,
                                             .68, 2)[0][0]
                origin = ((640 - text_width) // 2, 329)
                cv2.putText(frame, caption, origin, cv2.FONT_HERSHEY_SIMPLEX,
                            .68, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, caption, origin, cv2.FONT_HERSHEY_SIMPLEX,
                            .68, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(frame)
        writer.release()
        subprocess.run([str(FFMPEG), "-hide_banner", "-loglevel", "error",
                        "-nostdin", "-i", str(silent), "-f", "lavfi",
                        "-i", "sine=frequency=440:duration=3", "-c:v", "libx264",
                        "-crf", "17", "-c:a", "aac", "-shortest", str(output)],
                       check=True)
    print(output)


if __name__ == "__main__":
    main()
