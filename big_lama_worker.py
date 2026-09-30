"""Optional CUDA Big-LaMa worker for SubTitleRemover's binary frame protocol."""
from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from simple_lama_inpainting import SimpleLama


def read_exact(length: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < length:
        part = sys.stdin.buffer.read(length - len(chunks))
        if not part:
            raise EOFError("Incomplete frame from parent process")
        chunks.extend(part)
    return bytes(chunks)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: big_lama_worker.py MODEL_PATH")
    model_path = Path(sys.argv[1]).resolve(strict=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for the Big-LaMa worker")
    os.environ["LAMA_MODEL"] = str(model_path)
    protocol_output = sys.stdout.buffer
    sys.stdout = sys.stderr
    model = SimpleLama()
    protocol_output.write(b"READY\n")
    protocol_output.flush()
    while True:
        header = sys.stdin.buffer.read(8)
        if not header:
            break
        if len(header) != 8:
            raise EOFError("Incomplete frame header")
        width, height = struct.unpack("<II", header)
        if width <= 0 or height <= 0 or width * height > 16384 * 16384:
            raise ValueError("Invalid frame dimensions")
        bgr = np.frombuffer(read_exact(width * height * 3), np.uint8).reshape(
            height, width, 3)
        mask = np.frombuffer(read_exact(width * height), np.uint8).reshape(
            height, width)
        image = Image.fromarray(bgr[:, :, ::-1].copy(), "RGB")
        mask_image = Image.fromarray(mask.copy(), "L")
        repaired = model(image, mask_image)
        final = Image.composite(repaired, image, mask_image)
        output = np.asarray(final, np.uint8)[:, :, ::-1].copy()
        protocol_output.write(output.tobytes())
        protocol_output.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
