"""Restore bottom subtitles with same-shot clean frames and a local LaMa fallback.

Inputs are never changed.  The pass-one frame cache is private to the caller's
work directory, and the output is written as a new H.264/AAC-or-copy file.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
from time import perf_counter

import cv2
import numpy as np
import onnxruntime as ort
from rapidocr import RapidOCR


ROOT = (Path(sys.executable).resolve().parent if getattr(sys, "frozen", False)
        else Path(__file__).resolve().parent)
WORK_ROOT = ROOT / "work" / "subtitle"
FFMPEG = ROOT / "third_party" / "ffmpeg9" / "ffmpeg-9.0.1-essentials_build" / "bin" / "ffmpeg.exe"
FFPROBE = FFMPEG.with_name("ffprobe.exe")
MODEL = ROOT / "third_party" / "models" / "inpainting_lama_2025jan.onnx"
BIG_LAMA_WORKER = ROOT / "big_lama_worker.py"
Box = tuple[int, int, int, int]


@dataclass
class FrameInfo:
    boxes: list[Box]
    scene: int
    thumbnail: np.ndarray
    clean: bool = False


def frame_boxes(detector: RapidOCR, frame: np.ndarray,
                band_start: float) -> list[Box]:
    height, width = frame.shape[:2]
    boxes: list[Box] = []

    def collect(image: np.ndarray, scale: int = 1) -> None:
        result = detector(image)
        polygons = [] if result.boxes is None else result.boxes
        for polygon in polygons:
            left = max(0, int(np.min(polygon[:, 0]) / scale) - 5)
            right = min(width, int(np.max(polygon[:, 0]) / scale) + 6)
            top = max(0, int(np.min(polygon[:, 1]) / scale) - 5)
            bottom = min(height, int(np.max(polygon[:, 1]) / scale) + 6)
            middle = (left + right) / 2
            # A caption is a sufficiently wide, lower-centred overlay. This
            # deliberately leaves small labels and off-centre scene text alone.
            if (top < height * band_start or right - left < width * 0.08
                    or bottom - top < 8
                    or abs(middle - width / 2) > width * 0.28):
                continue
            existing = next((index for index, old in enumerate(boxes)
                             if abs((old[1] + old[3]) / 2
                                    - (top + bottom) / 2) < 14), None)
            if existing is None:
                boxes.append((left, top, right, bottom))
            else:
                old = boxes[existing]
                boxes[existing] = (min(old[0], left), min(old[1], top),
                                   max(old[2], right), max(old[3], bottom))

    collect(frame)
    if len(boxes) < 2:
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 2)
        collect(binary)
    if len(boxes) < 2:
        # Small outlined captions can disappear at source resolution. Detect
        # on enlarged pixels, then map the polygons back to the source frame.
        enlarged = cv2.resize(frame, None, fx=3, fy=3,
                              interpolation=cv2.INTER_CUBIC)
        collect(enlarged, 3)
        if len(boxes) < 2:
            enlarged_binary = cv2.adaptiveThreshold(
                cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY), 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 31, 4)
            collect(enlarged_binary, 3)
    return boxes


def box_mask(shape: tuple[int, ...], boxes: list[Box]) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for left, top, right, bottom in boxes:
        mask[top:bottom, left:right] = 255
    return mask


def restoration_mask(shape: tuple[int, ...], boxes: list[Box],
                     margin: int | None = None) -> np.ndarray:
    """Cover OCR's uncertain glyph edges and outlines before inpainting."""
    if margin is None:
        margin = max(6, round(shape[0] * 24 / 360))
    mask = box_mask(shape, boxes)
    if margin:
        mask = cv2.dilate(mask, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1)))
    return mask


def glyph_mask(frame: np.ndarray, boxes: list[Box]) -> np.ndarray:
    """Limit restoration to bright caption strokes and their dark outlines."""
    height, width = frame.shape[:2]
    region = np.zeros((height, width), dtype=np.uint8)
    for left, top, right, bottom in boxes:
        region[max(0, top - 2):min(height, bottom + 2),
               max(0, left - 2):min(width, right + 2)] = 1
    if not np.any(region):
        return region
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    saturation = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1]
    local_minimum = cv2.erode(gray, np.ones((5, 5), np.uint8))
    # A fixed dark threshold misses white captions over a light background:
    # their outlines can remain gray after compression. Require local contrast
    # as an alternative to a truly dark neighbouring pixel.
    dark_neighbour = ((local_minimum < 95)
                      | (gray.astype(np.int16) - local_minimum >= 55))
    strokes = ((gray >= 175) & (saturation <= 145) & dark_neighbour
               & (region > 0)).astype(np.uint8)
    diameter = max(7, 2 * round(height * 4 / 360) + 1)
    return cv2.dilate(strokes * 255,
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                (diameter, diameter)))


def expanded_refinement_mask(mask: np.ndarray) -> np.ndarray:
    """Include compression halos around detected glyphs for a final pass."""
    radius = max(2, round(mask.shape[0] * 12 / 360))
    return cv2.dilate(mask, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)))


def residual_ratio(restored: np.ndarray, boxes: list[Box],
                   original_mask: np.ndarray) -> float:
    total = int(np.count_nonzero(original_mask))
    if not boxes or total == 0:
        return 0.0
    return float(np.count_nonzero(glyph_mask(restored, boxes)) / total)


def thumbnail(frame: np.ndarray) -> np.ndarray:
    upper = frame[:int(frame.shape[0] * 0.68)]
    return cv2.resize(upper, (160, 60), interpolation=cv2.INTER_AREA)


def mean_difference(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.mean(cv2.absdiff(first, second)))


def detect_and_cache(source: Path, work: Path, start_frame: int,
                     end_frame: int, band_start: float,
                     scene_threshold: float) -> tuple[np.memmap, list[FrameInfo], float]:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open source video: {source}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0 or end_frame <= start_frame:
        raise RuntimeError("Invalid source timing or dimensions")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_count = end_frame - start_frame
    frames = np.memmap(work / "frames.bgr", dtype=np.uint8, mode="w+",
                       shape=(frame_count, height, width, 3))
    detector = RapidOCR()
    logging.getLogger("RapidOCR").setLevel(logging.ERROR)
    information: list[FrameInfo] = []
    scene = 0
    prior: np.ndarray | None = None
    started = perf_counter()
    try:
        for local_index in range(frame_count):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Decode ended at frame {start_frame + local_index}")
            frames[local_index] = frame
            small = thumbnail(frame)
            if prior is not None and mean_difference(small, prior) >= scene_threshold:
                scene += 1
            prior = small
            boxes = frame_boxes(detector, frame, band_start)
            information.append(FrameInfo(boxes, scene, small))
            if (local_index + 1) % 75 == 0 or local_index + 1 == frame_count:
                print(f"[detect] {local_index + 1}/{frame_count} frames; "
                      f"{scene + 1} scenes; {perf_counter() - started:.1f}s", flush=True)
    finally:
        capture.release()
        frames.flush()

    # OCR can miss a whole caption or one line for several consecutive frames.
    # Reuse a neighbouring line only when those pixels still match closely.
    detected = [item.boxes.copy() for item in information]
    for index, item in enumerate(information):
        if len(item.boxes) >= 2:
            continue
        frame = frames[index]
        for distance in range(1, max(2, int(round(fps * 0.8))) + 1):
            for neighbour_index in (index - distance, index + distance):
                if not 0 <= neighbour_index < frame_count:
                    continue
                neighbour = information[neighbour_index]
                if neighbour.scene != item.scene:
                    continue
                for candidate in detected[neighbour_index]:
                    centre = (candidate[1] + candidate[3]) / 2
                    if any(abs((box[1] + box[3]) / 2 - centre) < 14
                           for box in item.boxes):
                        continue
                    mask = box_mask(frame.shape, [candidate])
                    difference = cv2.absdiff(frame, frames[neighbour_index])
                    if float(np.mean(difference[mask > 0])) <= 4.5:
                        item.boxes.append(candidate)
                if len(item.boxes) >= 2:
                    break
            if len(item.boxes) >= 2:
                break
        if not item.boxes:
            item.clean = True
    return frames, information, fps


def candidate_quality(target: np.ndarray, reference: np.ndarray,
                      mask: np.ndarray) -> float:
    ring = cv2.subtract(
        cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))),
        mask,
    )
    if not np.any(ring):
        return float("inf")
    return float(np.mean(cv2.absdiff(target, reference)[ring > 0]))


def blend(target: np.ndarray, reference: np.ndarray, mask: np.ndarray,
          sigma: float = 2.0) -> np.ndarray:
    alpha = cv2.GaussianBlur(mask, (0, 0), sigma).astype(np.float32) / 255.0
    # Keep the interior fully replaced; only the border should feather.
    interior = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    alpha[interior > 0] = 1.0
    return np.clip(target.astype(np.float32) * (1 - alpha[..., None])
                   + reference.astype(np.float32) * alpha[..., None],
                   0, 255).astype(np.uint8)


class NeuralFallback:
    def __init__(self, model: Path):
        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(model), sess_options=options, providers=["CPUExecutionProvider"])
        if self.session.get_providers()[0] != "CPUExecutionProvider":
            raise RuntimeError("Unexpected inpainting provider")
        self.calls = 0

    def fill(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return frame.copy()
        height, width = frame.shape[:2]
        left = max(0, int(xs.min()) - 48)
        right = min(width, int(xs.max()) + 49)
        top = max(0, int(ys.min()) - 72)
        bottom = min(height, int(ys.max()) + 25)
        crop = frame[top:bottom, left:right]
        crop_mask = mask[top:bottom, left:right]
        crop_height, crop_width = crop.shape[:2]
        scale = min(1.0, 512 / crop_width, 512 / crop_height)
        if scale < 1:
            scaled_width = max(1, int(round(crop_width * scale)))
            scaled_height = max(1, int(round(crop_height * scale)))
            crop = cv2.resize(crop, (scaled_width, scaled_height),
                              interpolation=cv2.INTER_AREA)
            crop_mask = cv2.resize(crop_mask, (scaled_width, scaled_height),
                                   interpolation=cv2.INTER_NEAREST)
        small_height, small_width = crop.shape[:2]
        pad_left = (512 - small_width) // 2
        pad_right = 512 - small_width - pad_left
        pad_top = (512 - small_height) // 2
        pad_bottom = 512 - small_height - pad_top
        image_input = cv2.copyMakeBorder(
            crop, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_REFLECT_101)
        mask_input = cv2.copyMakeBorder(
            crop_mask, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=0)
        image_blob = (image_input.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        mask_blob = (mask_input.astype(np.float32) / 255.0)[None, None]
        output = self.session.run(None, {"image": image_blob, "mask": mask_blob})[0]
        output = np.clip(output[0].transpose(1, 2, 0), 0, 255).astype(np.uint8)
        output = output[pad_top:pad_top + small_height,
                        pad_left:pad_left + small_width]
        if output.shape[:2] != (crop_height, crop_width):
            output = cv2.resize(output, (crop_width, crop_height),
                                interpolation=cv2.INTER_CUBIC)
        result = frame.copy()
        result[top:bottom, left:right] = blend(
            frame[top:bottom, left:right], output,
            mask[top:bottom, left:right], sigma=1.8)
        self.calls += 1
        return result


class BigLamaWorker:
    """Keep the optional CUDA model loaded across frames in a separate Python."""

    def __init__(self, python: Path, model: Path, work: Path):
        if not BIG_LAMA_WORKER.is_file():
            raise FileNotFoundError(BIG_LAMA_WORKER)
        self.log = (work / "big-lama-worker.log").open("wb")
        self.process = subprocess.Popen(
            [str(python), str(BIG_LAMA_WORKER), str(model)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            bufsize=0)
        self.calls = 0
        if self.process.stdout is None or self.process.stdout.readline() != b"READY\n":
            self.close()
            raise RuntimeError(f"Big-LaMa worker failed; inspect {work / 'big-lama-worker.log'}")

    def fill(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("Big-LaMa worker pipes are unavailable")
        height, width = frame.shape[:2]
        payload = (struct.pack("<II", width, height)
                   + np.ascontiguousarray(frame).tobytes()
                   + np.ascontiguousarray(mask).tobytes())
        view = memoryview(payload)
        while view:
            sent = self.process.stdin.write(view)
            if sent is None or sent <= 0:
                raise RuntimeError("Big-LaMa worker stopped while receiving a frame")
            view = view[sent:]
        needed = width * height * 3
        received = bytearray()
        while len(received) < needed:
            piece = self.process.stdout.read(needed - len(received))
            if not piece:
                raise RuntimeError("Big-LaMa worker stopped while restoring a frame")
            received.extend(piece)
        self.calls += 1
        return np.frombuffer(received, np.uint8).reshape(height, width, 3).copy()

    def close(self) -> None:
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        if self.process.stdout is not None and not self.process.stdout.closed:
            self.process.stdout.close()
        if self.process.poll() is None:
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=10)
        self.log.close()


def create_inpainter(args: argparse.Namespace,
                     method: str) -> NeuralFallback | BigLamaWorker:
    if method == "gpu":
        return BigLamaWorker(args.big_lama_python, args.big_lama_model, args.work)
    return NeuralFallback(MODEL)


def load_strategy_map(path: Path | None, fps: float, total: int,
                      default: str) -> tuple[str, list[tuple[int, int, str]]]:
    """Convert non-overlapping subtitle strategy intervals to frame indices."""
    if path is None:
        return default, []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("segments", []), list):
        raise ValueError("Strategy map must contain a segments array")
    default = data.get("default", default)
    if default not in ("cpu", "gpu"):
        raise ValueError("Strategy map default must be cpu or gpu")
    spans = []
    for entry in data.get("segments", []):
        if not isinstance(entry, dict) or entry.get("method") not in ("cpu", "gpu"):
            raise ValueError("Every strategy segment needs method cpu or gpu")
        start, end = entry.get("start"), entry.get("end")
        if (not isinstance(start, (int, float)) or isinstance(start, bool)
                or not isinstance(end, (int, float)) or isinstance(end, bool)
                or not math.isfinite(start) or not math.isfinite(end)):
            raise ValueError("Strategy segment times must be finite numbers")
        first, last = round(start * fps), round(end * fps)
        if not 0 <= first < last <= total:
            raise ValueError("Strategy segment is outside the video or empty")
        spans.append((first, last, entry["method"]))
    spans.sort()
    if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
        raise ValueError("Strategy segments overlap")
    return default, spans


def load_mask_map(path: Path | None, fps: float, total: int,
                  width: int, height: int) -> list[tuple[int, int, list[Box]]]:
    """Load frame-aligned boxes that supplement OCR where a caption was missed."""
    if path is None:
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("segments"), list):
        raise ValueError("Mask map must contain a segments array")
    spans = []
    for entry in data["segments"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("boxes"), list):
            raise ValueError("Every mask segment needs a boxes array")
        start, end = entry.get("start"), entry.get("end")
        if (not isinstance(start, (int, float)) or isinstance(start, bool)
                or not isinstance(end, (int, float)) or isinstance(end, bool)
                or not math.isfinite(start) or not math.isfinite(end)):
            raise ValueError("Mask segment times must be finite numbers")
        first, last = round(start * fps), round(end * fps)
        if not 0 <= first < last <= total:
            raise ValueError("Mask segment is outside the video or empty")
        boxes = []
        for coordinates in entry["boxes"]:
            if (not isinstance(coordinates, list) or len(coordinates) != 4
                    or any(not isinstance(value, int) or isinstance(value, bool)
                           for value in coordinates)):
                raise ValueError("Mask boxes need four integer coordinates")
            left, top, right, bottom = coordinates
            if not (0 <= left < right <= width and 0 <= top < bottom <= height):
                raise ValueError("Mask box is outside the video or empty")
            boxes.append((left, top, right, bottom))
        if not boxes:
            raise ValueError("Mask segment must contain at least one box")
        spans.append((first, last, boxes))
    return spans


def supplement_boxes(information: list[FrameInfo], scan_start: int,
                     spans: list[tuple[int, int, list[Box]]]) -> None:
    """Add only the specified frames to the caption mask and reference exclusion."""
    scan_end = scan_start + len(information)
    for first, last, boxes in spans:
        for absolute in range(max(first, scan_start), min(last, scan_end)):
            item = information[absolute - scan_start]
            item.boxes.extend(boxes)
            item.clean = False


def strategy_for_frame(index: int, default: str,
                       spans: list[tuple[int, int, str]]) -> str:
    for first, last, method in spans:
        if first <= index < last:
            return method
    return default


def pick_reference(index: int, target: np.ndarray, mask: np.ndarray,
                   frames: np.memmap, information: list[FrameInfo],
                   candidate_indices: list[int], max_age: int) -> tuple[int | None, float]:
    item = information[index]
    choices = [other for other in candidate_indices
               if abs(index - other) <= max_age
               and information[other].scene == item.scene]
    choices.sort(key=lambda other: (
        mean_difference(item.thumbnail, information[other].thumbnail)
        + abs(index - other) * 0.002))
    best_index = None
    best_score = float("inf")
    for other in choices[:10]:
        upper_difference = mean_difference(
            item.thumbnail, information[other].thumbnail)
        if upper_difference > 5.0:
            continue
        score = candidate_quality(target, frames[other], mask)
        if score < best_score:
            best_index, best_score = other, score
    if best_score <= 3.0:
        return best_index, best_score
    return None, best_score


def pick_restored_reference(index: int, target: np.ndarray, mask: np.ndarray,
                            information: list[FrameInfo],
                            synthetic: dict[int, np.ndarray]) -> int | None:
    """Reuse a nearby restored patch when its shot and surrounding pixels match.

    The restoration engine is irrelevant here: a clean CPU patch can repair a
    later GPU frame (or vice versa) without a visible mid-caption transition.
    """
    item = information[index]
    for other in sorted(synthetic, key=lambda candidate: abs(index - candidate))[:12]:
        if information[other].scene != item.scene:
            continue
        if mean_difference(item.thumbnail, information[other].thumbnail) > 5:
            continue
        if candidate_quality(target, synthetic[other], mask) <= 3:
            return other
    return None


def probe_media(path: Path) -> dict:
    command = [str(FFPROBE), "-v", "error", "-show_streams", "-show_format",
               "-of", "json", str(path)]
    result = subprocess.run(command, capture_output=True, text=True,
                            check=True, encoding="utf-8", errors="replace")
    return json.loads(result.stdout)


def remove_private_work(work: Path) -> None:
    root = WORK_ROOT.resolve(strict=True)
    candidate = work.resolve(strict=True)
    if (work.is_symlink() or candidate.parent != root
            or not work.name.startswith("job-")
            or not (work / ".subtitle-work").is_file()):
        raise RuntimeError(f"Unsafe work directory: {work}")
    shutil.rmtree(work)


def validate_capture(path: Path, duration: float, frames: int,
                     audio_required: bool) -> None:
    data = probe_media(path)
    video = next((stream for stream in data["streams"]
                  if stream.get("codec_type") == "video"), None)
    if video is None or int(video.get("nb_frames", 0)) != frames:
        raise RuntimeError("Restored video has a missing or incorrect frame count")
    if abs(float(data["format"]["duration"]) - duration) > 0.1:
        raise RuntimeError("Restored video duration does not match the requested range")
    if audio_required and not any(stream.get("codec_type") == "audio"
                                  for stream in data["streams"]):
        raise RuntimeError("Restored video lost its audio stream")
    subprocess.run([str(FFMPEG), "-hide_banner", "-v", "error", "-xerror",
                    "-nostdin", "-i", str(path), "-f", "null", "NUL"],
                   check=True, capture_output=True)


def run_job(args: argparse.Namespace) -> dict:
    source = args.input.resolve(strict=True)
    output = args.output.resolve(strict=False)
    work = args.work.resolve(strict=False)
    if source == output or output.exists():
        raise ValueError("Output must be a new file, distinct from input")
    if output.suffix.lower() != ".mp4":
        raise ValueError("Output filename must end in .mp4")
    if not FFMPEG.is_file() or not FFPROBE.is_file():
        raise RuntimeError("Local FFmpeg is missing")
    if work.exists() and any(path.name != ".subtitle-work" for path in work.iterdir()):
        raise ValueError("Work directory must be new or empty")
    work.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if fps <= 0 or total < 2:
        raise ValueError("Input has no valid frame rate or frame count")
    mask_spans = load_mask_map(args.mask_map, fps, total, width, height)
    default_method, strategy_spans = load_strategy_map(
        args.strategy_map, fps, total,
        "gpu" if args.big_lama_python else "cpu")
    methods = {default_method, *(span[2] for span in strategy_spans)}
    if "gpu" in methods and not args.big_lama_python:
        raise ValueError("GPU strategy requires Big-LaMa Python and model")
    if "cpu" in methods and not MODEL.is_file():
        raise RuntimeError("Local CPU inpainting model is missing")
    start = int(round(args.start * fps))
    end = int(round((args.end or (total / fps)) * fps))
    if not 0 <= start < end <= total or args.band_start < 0 or args.band_start >= 1:
        raise ValueError("Invalid time range or subtitle band")
    audio_stream = next((stream for stream in probe_media(source)["streams"]
                         if stream.get("codec_type") == "audio"), None)
    audio_required = audio_stream is not None
    context = int(round(args.reference_seconds * fps))
    scan_start, scan_end = max(0, start - context), min(total, end + context)
    frames, information, measured_fps = detect_and_cache(
        source, work, scan_start, scan_end, args.band_start,
        args.scene_threshold)
    supplement_boxes(information, scan_start, mask_spans)
    if abs(measured_fps - fps) > 0.01:
        raise RuntimeError("Video frame rate changed during scan")
    selected_start, selected_end = start - scan_start, end - scan_start
    clean_candidates = [i for i, item in enumerate(information) if item.clean]
    synthetic: dict[int, np.ndarray] = {}
    synthetic_methods: dict[int, str] = {}
    models: dict[str, NeuralFallback | BigLamaWorker] = {}
    stats = {"source_frames": selected_end - selected_start,
             "scanned_frames": len(information), "clean_candidates": len(clean_candidates),
             "unchanged_frames": 0, "temporal_frames": 0,
             "synthetic_reuse_frames": 0, "neural_frames": 0,
             "cross_engine_reuse_frames": 0,
             "refined_frames": 0, "cpu_model_frames": 0,
             "gpu_model_frames": 0,
             "scenes": max(item.scene for item in information) + 1}
    log = (work / "ffmpeg.log").open("wb")
    duration = (end - start) / fps
    staged = work / "restored.partial.mp4"
    command = [str(FFMPEG), "-hide_banner", "-loglevel", "warning", "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{frames.shape[2]}x{frames.shape[1]}",
               "-r", str(fps), "-i", "pipe:0",
               "-ss", str(start / fps), "-t", str(duration), "-i", str(source),
               "-map", "0:v:0", "-map", "1:a:0?",
               "-frames:v", str(end - start), "-c:v", "libx264",
               "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart"]
    if audio_required:
        if audio_stream.get("codec_name") == "aac":
            command.extend(("-c:a", "copy"))
        else:
            command.extend(("-c:a", "aac", "-b:a", "192k"))
    command.append(str(staged))
    process = subprocess.Popen(command, stdin=subprocess.PIPE,
                               stdout=subprocess.DEVNULL, stderr=log)
    started = perf_counter()
    try:
        for index in range(selected_start, selected_end):
            original = np.asarray(frames[index]).copy()
            item = information[index]
            method = strategy_for_frame(scan_start + index, default_method,
                                        strategy_spans)
            model = models.get(method)
            if not item.boxes:
                restored = original
                stats["unchanged_frames"] += 1
            else:
                mask = glyph_mask(original, item.boxes)
                if not np.any(mask):
                    restored = original
                    stats["unchanged_frames"] += 1
                else:
                    reference, score = pick_reference(
                        index, original, mask, frames, information,
                        clean_candidates, context)
                    if reference is not None:
                        restored = blend(original, frames[reference], mask)
                        stats["temporal_frames"] += 1
                    else:
                        chosen = pick_restored_reference(
                            index, original, mask, information, synthetic)
                        if chosen is not None:
                            wide = expanded_refinement_mask(mask)
                            use_wide = (candidate_quality(
                                original, synthetic[chosen], wide) <= 3.0)
                            restored = blend(original, synthetic[chosen],
                                             wide if use_wide else mask)
                            stats["synthetic_reuse_frames"] += 1
                            if synthetic_methods[chosen] != method:
                                stats["cross_engine_reuse_frames"] += 1
                        else:
                            if model is None:
                                model = create_inpainter(args, method)
                                models[method] = model
                            # A second pass reduces high-contrast imprints.
                            restored = model.fill(model.fill(original, mask), mask)
                            synthetic[index] = restored
                            synthetic_methods[index] = method
                            stats["neural_frames"] += 1
                            stats[f"{method}_model_frames"] += 1
                    if residual_ratio(restored, item.boxes, mask) >= 0.45:
                        if model is None:
                            model = create_inpainter(args, method)
                            models[method] = model
                        restored = model.fill(
                            restored, expanded_refinement_mask(mask))
                        synthetic[index] = restored
                        synthetic_methods[index] = method
                        stats["refined_frames"] += 1
            if process.stdin is None:
                raise RuntimeError("FFmpeg input pipe is unavailable")
            process.stdin.write(restored.tobytes())
            progress = index - selected_start + 1
            if progress % 75 == 0 or progress == selected_end - selected_start:
                print(f"[restore] {progress}/{selected_end - selected_start}; "
                      f"{stats['temporal_frames']} source references, "
                      f"{stats['neural_frames']} model frames; "
                      f"{perf_counter() - started:.1f}s", flush=True)
        if process.stdin is not None:
            process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError(f"FFmpeg failed; inspect {work / 'ffmpeg.log'}")
    except BaseException:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        process.terminate()
        process.wait(timeout=10)
        raise
    finally:
        log.close()
        for active_model in models.values():
            if isinstance(active_model, BigLamaWorker):
                active_model.close()
        frames._mmap.close()
        del frames
    validate_capture(staged, duration, end - start, audio_required)
    os.rename(staged, output)
    stats.update({"input": str(source), "output": str(output),
                  "start_seconds": start / fps, "end_seconds": end / fps,
                  "fps": fps,
                  "engine": ("same-shot-reference+mapped-LaMa-engines"
                             if args.strategy_map else
                             "same-shot-reference+Big-LaMa-CUDA"
                             if args.big_lama_python else
                             "same-shot-reference+OpenCV-LaMa-ONNX-CPU")})
    (work / "restore-report.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats


def run_chunked_job(args: argparse.Namespace) -> dict:
    """Restore frame-aligned pieces with context, then remux original audio once."""
    source = args.input.resolve(strict=True)
    output = args.output.resolve(strict=False)
    if source == output or output.exists() or output.suffix.lower() != ".mp4":
        raise ValueError("Output must be a new MP4 distinct from input")
    capture = cv2.VideoCapture(str(source))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if fps <= 0 or total < 2 or args.chunk_seconds <= 0:
        raise ValueError("Invalid source timing or chunk duration")
    start = int(round(args.start * fps))
    end = int(round((args.end or (total / fps)) * fps))
    chunk_frames = int(round(args.chunk_seconds * fps))
    if not 0 <= start < end <= total or chunk_frames < 2:
        raise ValueError("Invalid range or chunk size")
    pieces: list[tuple[Path, int]] = []
    reports: list[dict] = []
    for number, first in enumerate(range(start, end, chunk_frames)):
        last = min(end, first + chunk_frames)
        piece = args.work / f"chunk-{number:04d}.mp4"
        piece_args = argparse.Namespace(**vars(args))
        piece_args.start = first / fps
        piece_args.end = last / fps
        piece_args.output = piece
        piece_args.work = args.work / f"chunk-{number:04d}-work"
        reports.append(run_job(piece_args))
        # The concat demuxer offsets video timestamps when the intermediate
        # MP4s contain AAC edit lists. Keep only video in the pieces; original
        # audio is mapped once at the final remux below.
        video_piece = args.work / f"video-{number:04d}.mp4"
        subprocess.run([str(FFMPEG), "-hide_banner", "-loglevel", "error",
                        "-y", "-i", str(piece), "-map", "0:v:0",
                        "-c:v", "copy", "-an", str(video_piece)],
                       check=True, capture_output=True)
        pieces.append((video_piece, last - first))
        print(f"[chunk] {number + 1} complete: frames {first}-{last - 1}",
              flush=True)

    manifest = args.work / "pieces.ffconcat"
    manifest.write_text(
        "ffconcat version 1.0\n" + "".join(
            f"file '{piece.name}'\nduration {frames / fps:.9f}\n"
            for piece, frames in pieces),
        encoding="utf-8")
    input_info = probe_media(source)
    audio = next((stream for stream in input_info["streams"]
                  if stream.get("codec_type") == "audio"), None)
    duration = (end - start) / fps
    staged = args.work / "joined.partial.mp4"
    command = [str(FFMPEG), "-hide_banner", "-loglevel", "warning", "-y",
               "-f", "concat", "-safe", "0", "-i", str(manifest),
               "-ss", str(start / fps), "-t", str(duration),
               "-i", str(source), "-map", "0:v:0", "-map", "1:a:0?",
               "-frames:v", str(end - start), "-c:v", "copy"]
    if audio is not None:
        command.extend(("-c:a", "copy") if audio.get("codec_name") == "aac"
                       else ("-c:a", "aac", "-b:a", "192k"))
    command.extend(("-movflags", "+faststart", str(staged)))
    with (args.work / "concat-ffmpeg.log").open("wb") as log:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL,
                       stderr=log)
    validate_capture(staged, duration, end - start, audio is not None)
    os.rename(staged, output)
    keys = ("unchanged_frames", "temporal_frames", "synthetic_reuse_frames",
            "cross_engine_reuse_frames",
            "neural_frames", "refined_frames", "cpu_model_frames",
            "gpu_model_frames")
    return {"input": str(source), "output": str(output),
            "start_seconds": start / fps, "end_seconds": end / fps,
            "fps": fps, "source_frames": end - start,
            "chunk_seconds": args.chunk_seconds,
            "chunks": len(pieces), "engine": "frame-aligned-chunks+" +
            ("same-shot-reference+mapped-LaMa-engines" if args.strategy_map else
             "same-shot-reference+Big-LaMa-CUDA" if args.big_lama_python else
             "same-shot-reference+OpenCV-LaMa-ONNX-CPU"),
            **{key: sum(report[key] for report in reports) for key in keys}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--work", type=Path,
                        help="Optional diagnostic directory; retained after processing")
    parser.add_argument("--keep-work", action="store_true",
                        help="Retain automatically created temporary files")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=0.0)
    parser.add_argument("--reference-seconds", type=float, default=8.0)
    parser.add_argument("--band-start", type=float, default=0.64)
    parser.add_argument("--scene-threshold", type=float, default=20.0)
    parser.add_argument("--chunk-seconds", type=float, default=0.0,
                        help="Process independent frame-aligned chunks with context")
    parser.add_argument("--big-lama-python", type=Path,
                        help="Python with CUDA torch and simple-lama-inpainting")
    parser.add_argument("--big-lama-model", type=Path,
                        help="Local big-lama.pt used by the CUDA worker")
    parser.add_argument("--strategy-map", type=Path,
                        help="JSON with default cpu/gpu and non-overlapping time spans")
    parser.add_argument("--mask-map", type=Path,
                        help="JSON with timed pixel boxes to supplement missed OCR captions")
    args = parser.parse_args()
    if bool(args.big_lama_python) != bool(args.big_lama_model):
        parser.error("Provide both --big-lama-python and --big-lama-model")
    if args.big_lama_python:
        args.big_lama_python = args.big_lama_python.resolve(strict=True)
        args.big_lama_model = args.big_lama_model.resolve(strict=True)
    if args.strategy_map:
        args.strategy_map = args.strategy_map.resolve(strict=True)
    if args.mask_map:
        args.mask_map = args.mask_map.resolve(strict=True)
    logging.getLogger("RapidOCR").setLevel(logging.ERROR)
    automatic_work = args.work is None
    if automatic_work:
        WORK_ROOT.mkdir(parents=True, exist_ok=True)
        args.work = Path(tempfile.mkdtemp(prefix="job-", dir=WORK_ROOT))
        (args.work / ".subtitle-work").write_text(
            "subtitle restoration temporary data\n", encoding="utf-8")
    completed = False
    try:
        report = (run_chunked_job(args) if args.chunk_seconds else run_job(args))
        args.output.resolve(strict=True).with_suffix(".restoration.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        completed = True
    except Exception as error:
        print(f"Subtitle restoration failed: {error}", file=sys.stderr)
        return 1
    finally:
        if automatic_work and completed and not args.keep_work:
            try:
                remove_private_work(args.work)
            except OSError as error:
                print(f"Verified output is complete; temporary cleanup was deferred: {error}",
                      file=sys.stderr)
        elif automatic_work:
            print(f"Work retained: {args.work}", file=sys.stderr)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
