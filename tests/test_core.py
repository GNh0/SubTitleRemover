import unittest
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np

import remove_subtitles
from remove_subtitles import (BigLamaWorker, FrameInfo, blend, box_mask, candidate_quality,
                              expanded_refinement_mask, frame_boxes, glyph_mask,
                              load_strategy_map, pick_reference,
                              pick_restored_reference, residual_ratio,
                              restoration_mask, strategy_for_frame)


class SubtitleRestorationTests(unittest.TestCase):
    def test_matching_clean_frame_restores_only_masked_region(self):
        source = np.full((40, 60, 3), 100, dtype=np.uint8)
        subtitled = source.copy()
        subtitled[28:33, 18:42] = 245
        mask = box_mask(subtitled.shape, [(16, 26, 44, 35)])
        frames = np.stack([subtitled, source])
        info = [
            FrameInfo([(16, 26, 44, 35)], 0, np.full((4, 4, 3), 100, np.uint8)),
            FrameInfo([], 0, np.full((4, 4, 3), 100, np.uint8), True),
        ]
        reference, score = pick_reference(0, subtitled, mask, frames, info, [1], 5)
        self.assertEqual((reference, score), (1, 0))
        restored = blend(subtitled, frames[reference], mask)
        self.assertEqual(restored[30, 30, 0], 100)
        self.assertEqual(restored[2, 2, 0], 100)

    def test_changed_boundary_rejects_reference(self):
        source = np.zeros((40, 60, 3), dtype=np.uint8)
        changed = np.full_like(source, 60)
        mask = box_mask(source.shape, [(12, 25, 45, 34)])
        self.assertGreater(candidate_quality(source, changed, mask), 3)
        info = [
            FrameInfo([(12, 25, 45, 34)], 0, np.zeros((4, 4, 3), np.uint8)),
            FrameInfo([], 0, np.zeros((4, 4, 3), np.uint8), True),
        ]
        reference, _ = pick_reference(0, source, mask,
                                      np.stack([source, changed]), info, [1], 5)
        self.assertIsNone(reference)

    def test_other_scene_is_never_used(self):
        frame = np.zeros((40, 60, 3), dtype=np.uint8)
        mask = box_mask(frame.shape, [(12, 25, 45, 34)])
        info = [
            FrameInfo([(12, 25, 45, 34)], 0, np.zeros((4, 4, 3), np.uint8)),
            FrameInfo([], 1, np.zeros((4, 4, 3), np.uint8), True),
        ]
        reference, _ = pick_reference(0, frame, mask,
                                      np.stack([frame, frame]), info, [1], 5)
        self.assertIsNone(reference)

    def test_restored_patch_is_reused_across_engine_boundary_in_same_shot(self):
        clean = np.full((40, 60, 3), 100, dtype=np.uint8)
        subtitled = clean.copy()
        subtitled[28:33, 18:42] = 240
        mask = box_mask(subtitled.shape, [(16, 26, 44, 35)])
        matching = np.full((4, 4, 3), 100, dtype=np.uint8)
        info = [FrameInfo([], 0, matching),
                FrameInfo([(16, 26, 44, 35)], 0, matching)]
        # The earlier patch may have come from CPU even if this frame uses GPU.
        self.assertEqual(pick_restored_reference(
            1, subtitled, mask, info, {0: clean}), 0)
        info[0].scene = 1
        self.assertIsNone(pick_restored_reference(
            1, subtitled, mask, info, {0: clean}))
        info[0].scene = 0
        changed = np.full_like(clean, 40)
        self.assertIsNone(pick_restored_reference(
            1, subtitled, mask, info, {0: changed}))

    def test_mask_covers_caption_outline(self):
        mask = restoration_mask((360, 640, 3), [(265, 295, 394, 337)])
        self.assertEqual(mask[284, 255], 255)
        self.assertEqual(mask[220, 255], 0)
        large = restoration_mask((720, 1280, 3), [(530, 590, 788, 674)])
        self.assertEqual(large[568, 510], 255)

    def test_off_centre_scene_sign_is_not_a_subtitle(self):
        class Detection:
            def __init__(self, boxes):
                self.boxes = boxes

        def detector(image):
            scale = image.shape[1] / 640
            return Detection(np.array([
                [[25, 255], [180, 255], [180, 284], [25, 284]],
                [[185, 305], [455, 305], [455, 332], [185, 332]],
            ]) * scale)

        boxes = frame_boxes(detector,
                            np.zeros((360, 640, 3), np.uint8), .64)
        self.assertEqual(boxes, [(180, 300, 461, 338)])

    def test_enlarged_detection_recovers_small_caption(self):
        class Detection:
            def __init__(self, boxes):
                self.boxes = boxes

        def detector(image):
            if image.shape[1] == 640:
                return Detection(None)
            return Detection(np.array([
                [[780, 900], [1170, 900], [1170, 990], [780, 990]],
            ]))

        boxes = frame_boxes(detector,
                            np.zeros((360, 640, 3), np.uint8), .64)
        self.assertEqual(boxes, [(255, 295, 396, 336)])

    def test_enlarged_binary_recovers_second_line(self):
        class Detection:
            def __init__(self, boxes):
                self.boxes = boxes

        def detector(image):
            if image.shape[1] == 640:
                return Detection(None)
            if image.ndim == 3:
                return Detection(np.array([
                    [[660, 825], [1305, 825], [1305, 915], [660, 915]],
                ]))
            return Detection(np.array([
                [[540, 900], [1440, 900], [1440, 990], [540, 990]],
            ]))

        boxes = frame_boxes(detector,
                            np.zeros((360, 640, 3), np.uint8), .64)
        self.assertEqual(boxes, [(215, 270, 441, 311),
                                 (175, 295, 486, 336)])

    def test_glyph_mask_avoids_unrelated_light_pixels(self):
        frame = np.full((360, 640, 3), 110, np.uint8)
        frame[300:320, 260:390] = 245
        frame[302:318, 265:385] = 20
        frame[305:315, 270:380] = 245
        frame[300:320, 450:470] = 245
        mask = glyph_mask(frame, [(255, 295, 395, 330)])
        self.assertEqual(mask[309, 300], 255)
        self.assertEqual(mask[309, 455], 0)
        self.assertEqual(mask[200, 300], 0)

    def test_glyph_mask_finds_white_caption_on_light_background(self):
        frame = np.full((360, 640, 3), 140, np.uint8)
        frame[300:320, 260:390] = 135
        frame[304:316, 265:385] = 245
        mask = glyph_mask(frame, [(255, 295, 395, 325)])
        self.assertEqual(mask[309, 300], 255)
        self.assertEqual(mask[200, 300], 0)

    def test_residual_refinement_expands_only_detected_strokes(self):
        original = np.full((360, 640, 3), 110, np.uint8)
        original[300:320, 260:390] = 20
        original[304:316, 264:386] = 245
        boxes = [(255, 295, 395, 325)]
        mask = glyph_mask(original, boxes)
        self.assertEqual(residual_ratio(original, boxes, mask), 1.0)
        clean = np.full_like(original, 110)
        self.assertEqual(residual_ratio(clean, boxes, mask), 0.0)
        expanded = expanded_refinement_mask(mask)
        self.assertEqual(expanded[300, 254], 255)
        self.assertEqual(expanded[250, 254], 0)

    def test_big_lama_worker_transfers_frames_without_changing_them(self):
        worker_source = """import struct, sys
stream = sys.stdin.buffer
output = sys.stdout.buffer
output.write(b'READY\\n')
output.flush()
while header := stream.read(8):
    width, height = struct.unpack('<II', header)
    frame = stream.read(width * height * 3)
    stream.read(width * height)
    output.write(frame)
    output.flush()
"""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            script = root / "worker.py"
            script.write_text(worker_source, encoding="utf-8")
            with patch.object(remove_subtitles, "BIG_LAMA_WORKER", script):
                worker = BigLamaWorker(Path(sys.executable), root / "model.pt", root)
                try:
                    frame = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
                    mask = np.full((4, 5), 255, np.uint8)
                    np.testing.assert_array_equal(worker.fill(frame, mask), frame)
                    self.assertEqual(worker.calls, 1)
                finally:
                    worker.close()

    def test_strategy_map_selects_frame_aligned_non_overlapping_spans(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "strategies.json"
            path.write_text(json.dumps({
                "default": "gpu",
                "segments": [{"start": 2.0, "end": 4.0, "method": "cpu"}]
            }), encoding="utf-8")
            default, spans = load_strategy_map(path, 15, 150, "cpu")
            self.assertEqual((default, spans),
                             ("gpu", [(30, 60, "cpu")]))
            self.assertEqual(strategy_for_frame(29, default, spans), "gpu")
            self.assertEqual(strategy_for_frame(30, default, spans), "cpu")
            self.assertEqual(strategy_for_frame(59, default, spans), "cpu")
            self.assertEqual(strategy_for_frame(60, default, spans), "gpu")
            path.write_text(json.dumps({"segments": [
                {"start": 2, "end": 4, "method": "cpu"},
                {"start": 3, "end": 5, "method": "gpu"},
            ]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overlap"):
                load_strategy_map(path, 15, 150, "cpu")


if __name__ == "__main__":
    unittest.main()
