#!/usr/bin/env python3
"""
screen_chess_bot.py

A production-ready chess automation agent that reads an on-screen chess board,
keeps a python-chess game state in sync, consults Stockfish for best moves, and
performs mouse interactions via pyautogui.
"""

from __future__ import annotations

import argparse
import collections
import ctypes
import dataclasses
import enum
import io
import json
import logging
import math
import queue
import threading
import os
import random
import sys
import time
from pathlib import Path
from typing import Callable, Deque, Dict, Iterable, List, Optional, Sequence, Tuple, Set, Union

import chess  # type: ignore
import cv2  # type: ignore
import mss  # type: ignore
import numpy as np
import pyautogui  # type: ignore
import tkinter as tk
from tkinter import messagebox
from pydantic import BaseModel, ValidationInfo, field_validator
from stockfish import Stockfish  # type: ignore
from PIL import Image, ImageTk  # type: ignore

try:
    from stockfish.models import StockfishException  # type: ignore
except Exception:  # pragma: no cover - fallback for older versions
    StockfishException = Exception  # type: ignore


__version__ = "0.9.0"


BOARD_CANVAS_SIZE = 800
SQUARES_PER_SIDE = 8
SQUARE_MARGIN_RATIO = 0.08
STABILITY_HISTORY = 3
REQUIRED_STABLE_COUNT = 2
DEFAULT_PIECE_LABEL = "."
DEFAULT_WHITE_THRESHOLD = 0.68
DEFAULT_BLACK_THRESHOLD = 0.68
TEMPLATE_FILENAMES = {
    "wP": "wp.png",
    "wN": "wn.png",
    "wB": "wb.png",
    "wR": "wr.png",
    "wQ": "wq.png",
    "wK": "wk.png",
    "bP": "bp.png",
    "bN": "bn.png",
    "bB": "bb.png",
    "bR": "br.png",
    "bQ": "bq.png",
    "bK": "bk.png",
}
TEMPLATE_COLOR = (255, 255, 255)
TEMPLATE_BG = (0, 0, 0)
PROMOTION_MAP = {
    chess.QUEEN: "q",
    chess.ROOK: "r",
    chess.BISHOP: "b",
    chess.KNIGHT: "n",
}
ALERT_PROMPT_THRESHOLD = 50
ALERT_RESTART_THRESHOLD = 150
AMBIGUOUS_RESTART_THRESHOLD = 20
BOARD_COORDINATES_FILE = Path(__file__).resolve().parent / "board_coordinates.json"


class BotOrientation(enum.Enum):
    WHITE_BOTTOM = "white_bottom"
    BLACK_BOTTOM = "black_bottom"

    @property
    def is_white_bottom(self) -> bool:
        return self is BotOrientation.WHITE_BOTTOM

    @property
    def is_black_bottom(self) -> bool:
        return self is BotOrientation.BLACK_BOTTOM


@dataclasses.dataclass(frozen=True)
class ScreenTransform:
    bbox: Dict[str, int]
    src_points: np.ndarray
    dst_points: np.ndarray
    matrix: np.ndarray
    matrix_inv: np.ndarray
    canvas_size: Tuple[int, int]

    def warp(self, frame: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(frame, self.matrix, self.canvas_size, flags=cv2.INTER_LINEAR)

    def normalized_center_to_screen(self, center_xy: Tuple[float, float]) -> Tuple[float, float]:
        x, y = center_xy
        homogeneous = np.array([x, y, 1.0], dtype=np.float32)
        mapped = self.matrix_inv @ homogeneous
        mapped /= mapped[2]
        rel_x, rel_y = float(mapped[0]), float(mapped[1])
        screen_x = self.bbox["left"] + rel_x
        screen_y = self.bbox["top"] + rel_y
        return screen_x, screen_y


@dataclasses.dataclass
class SquareDetection:
    label: str
    score: float


@dataclasses.dataclass
class BoardDetection:
    grid: List[List[SquareDetection]]
    warped_bgr: np.ndarray
    clahe_gray: np.ndarray
    timestamp: float


@dataclasses.dataclass(frozen=True)
class BoardSelectionResult:
    points: List[Tuple[int, int]]
    save_requested: bool


class RestartRequested(Exception):
    def __init__(self, board_points: Sequence[Tuple[int, int]]) -> None:
        super().__init__("Restart requested due to persistent low-confidence detection.")
        self.points = list(board_points)


class CLIConfig(BaseModel):
    engine_path: Optional[str]
    move_time_ms: Optional[int]
    move_delay_range_s: Optional[Tuple[float, float]] = None
    stockfish_power_range: Optional[Tuple[int, int]] = None
    use_cached_coord: Optional[int] = None
    launch_ui: bool = False
    depth: int = 12
    color: str = "auto"
    show_debug: bool = False
    promotion: str = "q"
    yolo_weights: Optional[str] = None
    yolo_confidence: float = 0.35
    yolo_iou: float = 0.45
    yolo_device: Optional[str] = None
    trust_detections: bool = False
    show_move_only: bool = False
    single_move_hotkeys: bool = False

    @field_validator("color")
    @classmethod
    def validate_color(cls, value: str) -> str:
        allowed = {"auto", "white", "black"}
        lower = value.lower()
        if lower not in allowed:
            raise ValueError(f"--color must be one of {allowed}")
        return lower

    @field_validator("promotion")
    @classmethod
    def validate_promotion(cls, value: str) -> str:
        allowed = {"q", "r", "b", "n"}
        lower = value.lower()
        if lower not in allowed:
            raise ValueError(f"--promotion must be one of {allowed}")
        return lower

    @field_validator("move_time_ms")
    @classmethod
    def validate_move_time(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value <= 0:
            raise ValueError("--move-time-ms must be positive")
        return value

    @field_validator("move_delay_range_s")
    @classmethod
    def validate_move_delay_range(
        cls, value: Optional[Tuple[float, float]]
    ) -> Optional[Tuple[float, float]]:
        if value is None:
            return value
        low, high = value
        if low < 0 or high < 0:
            raise ValueError("--move-delay-range values must be non-negative")
        if high < low:
            raise ValueError("--move-delay-range max must be greater than or equal to min")
        return (float(low), float(high))

    @field_validator("stockfish_power_range")
    @classmethod
    def validate_stockfish_power_range(
        cls, value: Optional[Tuple[int, int]]
    ) -> Optional[Tuple[int, int]]:
        if value is None:
            return value
        low, high = value
        if not (0 <= low <= 20 and 0 <= high <= 20):
            raise ValueError("--stockfish-power values must be between 0 and 20.")
        if high < low:
            raise ValueError("--stockfish-power max must be greater than or equal to min.")
        return (int(low), int(high))

    @field_validator("use_cached_coord")
    @classmethod
    def validate_use_cached_coord(cls, value: Optional[int]) -> Optional[int]:
        if value is None:
            return value
        if value < 0:
            raise ValueError("--use-cached-coord must be a non-negative integer.")
        return int(value)

    @field_validator("depth")
    @classmethod
    def validate_depth(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("--depth must be positive")
        return value

    @field_validator("yolo_confidence", "yolo_iou")
    @classmethod
    def validate_probabilities(cls, value: float, info: ValidationInfo) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError(f"--{info.field_name.replace('_', '-')} must be between 0 and 1.")
        return value


@dataclasses.dataclass
class Template:
    label: str
    original: np.ndarray
    edges: np.ndarray


class YOLOPieceDetector:
    def __init__(
        self,
        weights_path: Path,
        confidence: float,
        iou: float,
        device: Optional[str] = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for YOLO-based detection. Install it via 'pip install ultralytics'."
            ) from exc

        resolved_path = Path(os.path.expanduser(str(weights_path))).resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(f"YOLO weights file not found at {resolved_path}")

        self.model = YOLO(str(resolved_path))
        self.confidence = confidence
        self.iou = iou
        self.device = device

        names = getattr(self.model, "names", None)
        if isinstance(names, dict):
            raw_map = {int(idx): label for idx, label in names.items()}
        elif isinstance(names, list):
            raw_map = {idx: label for idx, label in enumerate(names)}
        else:
            raise ValueError("YOLO model does not expose class names mapping.")

        if not raw_map:
            raise ValueError("YOLO model class mapping is empty. Check the training configuration.")

        self.class_to_label = raw_map
        self.class_to_canonical: Dict[int, str] = {}
        for cls_id, label in raw_map.items():
            canonical = self._to_canonical_label(label)
            if canonical is None:
                logging.warning("Skipping YOLO class '%s' (id=%s): unable to map to chess label.", label, cls_id)
                continue
            self.class_to_canonical[cls_id] = canonical

        if not self.class_to_canonical:
            raise ValueError(
                "None of the YOLO classes could be mapped to chess labels. "
                "Expected names like 'w-pawn' or 'b_knight'."
            )

        logging.info(
            "Loaded YOLO model from %s with classes: %s",
            resolved_path,
            ", ".join(sorted(raw_map.values())),
        )

    def detect(self, warped_bgr: np.ndarray) -> List[List[SquareDetection]]:
        grid: List[List[SquareDetection]] = [
            [SquareDetection(label=DEFAULT_PIECE_LABEL, score=0.0) for _ in range(SQUARES_PER_SIDE)]
            for _ in range(SQUARES_PER_SIDE)
        ]

        results = self.model.predict(
            warped_bgr,
            imgsz=BOARD_CANVAS_SIZE,
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        if not results:
            return grid

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return grid

        cell_size = BOARD_CANVAS_SIZE / SQUARES_PER_SIDE
        for box in boxes:
            cls_tensor = getattr(box, "cls", None)
            xyxy_tensor = getattr(box, "xyxy", None)
            if cls_tensor is None or xyxy_tensor is None:
                continue

            cls_id = int(cls_tensor.item())
            canonical_label = self.class_to_canonical.get(cls_id)
            if not canonical_label:
                continue

            conf_tensor = getattr(box, "conf", None)
            confidence = float(conf_tensor.item()) if conf_tensor is not None else 0.0
            coords = [float(v) for v in xyxy_tensor[0].tolist()]
            if len(coords) != 4:
                continue
            x1, y1, x2, y2 = coords
            center_x = (x1 + x2) * 0.5
            center_y = (y1 + y2) * 0.5
            col = int(center_x // cell_size)
            row = int(center_y // cell_size)
            if not (0 <= row < SQUARES_PER_SIDE and 0 <= col < SQUARES_PER_SIDE):
                continue

            existing = grid[row][col]
            if confidence > existing.score:
                grid[row][col] = SquareDetection(label=canonical_label, score=confidence)

        return grid

    @staticmethod
    def _to_canonical_label(raw_label: str) -> Optional[str]:
        cleaned = raw_label.strip().lower().replace(" ", "").replace("_", "-")
        if len(cleaned) == 2 and cleaned[0] in {"w", "b"}:
            return cleaned[0] + cleaned[1].upper()
        if "-" not in cleaned:
            return None
        color, piece = cleaned.split("-", 1)
        if not color:
            return None
        color_char = color[0]
        if color_char not in {"w", "b"}:
            return None
        piece_map = {
            "pawn": "P",
            "rook": "R",
            "knight": "N",
            "bishop": "B",
            "queen": "Q",
            "king": "K",
        }
        piece_char = piece_map.get(piece)
        if piece_char is None:
            return None
        return f"{color_char}{piece_char}"


def square_margin_pixels(cell_size: int) -> int:
    return int(round(cell_size * SQUARE_MARGIN_RATIO))


def ensure_channel_order(img: np.ndarray) -> np.ndarray:
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def create_clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def preprocess_edges(img_gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 110)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.erode(edges, np.ones((3, 3), np.uint8), iterations=1)
    return edges


def build_fen_from_grid(grid: List[List[str]], orientation: BotOrientation) -> str:
    rows: List[str] = []
    iterable = grid if orientation.is_white_bottom else list(reversed(grid))
    for row in iterable:
        fen_row = ""
        empty_run = 0
        for label in row:
            if label == DEFAULT_PIECE_LABEL:
                empty_run += 1
            else:
                if empty_run:
                    fen_row += str(empty_run)
                    empty_run = 0
                fen_row += label
        if empty_run:
            fen_row += str(empty_run)
        rows.append(fen_row)
    return "/".join(rows)


def normalize_piece_label(label: str) -> str:
    if len(label) != 2:
        return DEFAULT_PIECE_LABEL
    color, piece = label
    if color not in {"w", "b"}:
        return DEFAULT_PIECE_LABEL
    return piece.upper() if color == "w" else piece.lower()


def detection_grid_to_piece_map(
    detection: BoardDetection,
    orientation: BotOrientation,
) -> Dict[chess.Square, str]:
    piece_map: Dict[chess.Square, str] = {}
    for row_idx, row in enumerate(detection.grid):
        for col_idx, sq in enumerate(row):
            fen_symbol = normalize_piece_label(sq.label)
            if fen_symbol == DEFAULT_PIECE_LABEL:
                continue
            if orientation.is_white_bottom:
                file_idx = col_idx
                rank_idx = 7 - row_idx
            else:
                file_idx = 7 - col_idx
                rank_idx = row_idx
            square = chess.square(file_idx, rank_idx)
            piece_map[square] = fen_symbol
    return piece_map


def square_to_grid_indices(square: chess.Square, orientation: BotOrientation) -> Tuple[int, int]:
    file_idx = chess.square_file(square)
    rank_idx = chess.square_rank(square)
    if orientation.is_white_bottom:
        col = file_idx
        row = 7 - rank_idx
    else:
        col = 7 - file_idx
        row = rank_idx
    return row, col


def piece_to_detection_label(piece: chess.Piece) -> str:
    color = "w" if piece.color == chess.WHITE else "b"
    type_map = {
        chess.PAWN: "P",
        chess.ROOK: "R",
        chess.KNIGHT: "N",
        chess.BISHOP: "B",
        chess.QUEEN: "Q",
        chess.KING: "K",
    }
    symbol = type_map.get(piece.piece_type)
    if symbol is None:
        return DEFAULT_PIECE_LABEL
    return f"{color}{symbol}"


def console_key_pressed() -> Optional[str]:
    if os.name == "nt":
        import msvcrt  # type: ignore

        if msvcrt.kbhit():
            key = msvcrt.getwch()
            return key.lower()
        return None
    import select
    rlist, _, _ = select.select([sys.stdin], [], [], 0)
    if rlist:
        ch = sys.stdin.read(1)
        return ch.lower()
    return None


def prompt_match_end_confirmation(repetitions: int) -> Optional[bool]:
    message = (
        "Piece detection has stayed unstable for a while.\n"
        "Did the match end? Choose Yes to stop the bot or No to keep waiting."
    )
    root: Optional[tk.Tk] = None
    try:
        root = tk.Tk()
        root.withdraw()
        result = messagebox.askyesno(
            title="Chess Bot",
            message=f"{message}\n\nLow-confidence checks: {repetitions}",
        )
        return result
    except Exception:
        logging.exception("Failed to display confirmation dialog.")
        return None
    finally:
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass


def board_piece_map(board: chess.Board) -> Dict[chess.Square, str]:
    return {
        square: board.piece_at(square).symbol()
        for square in chess.SQUARES
        if board.piece_at(square) is not None
    }


def compare_piece_maps(
    expected: Dict[chess.Square, str],
    actual: Dict[chess.Square, str],
) -> int:
    diff = {sq for sq in chess.SQUARES if expected.get(sq) != actual.get(sq)}
    return len(diff)


def verify_expected_position(
    reader: BoardReader,
    orientation: BotOrientation,
    expected_map: Dict[chess.Square, str],
    attempts: int = 4,
    delay: float = 0.35,
    critical_squares: Optional[Iterable[chess.Square]] = None,
) -> bool:
    for _ in range(attempts):
        detection = reader.capture_warped()
        actual_map = detection_grid_to_piece_map(detection, orientation)
        if critical_squares is not None:
            remaining = [
                sq
                for sq in critical_squares
                if expected_map.get(sq) != actual_map.get(sq)
            ]
            if not remaining:
                return True
        mismatches = compare_piece_maps(expected_map, actual_map)
        if mismatches == 0:
            return True
        if mismatches <= 2:
            logging.debug("Position nearly matches expected (%d mismatches).", mismatches)
            return True
        time.sleep(delay)
    return False


class ScreenSelector:
    def __init__(self) -> None:
        self.points: List[Tuple[int, int]] = []

    def _mouse_callback(self, event: int, x: int, y: int, _flags: int, _userdata: Optional[object]) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 4:
                self.points.append((x, y))
                logging.debug("Captured point %s", self.points[-1])
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.points:
                self.points.pop()
                logging.info("Removed last point. %d points left.", len(self.points))

    def _draw_overlay(self, base_img: np.ndarray) -> np.ndarray:
        overlay = base_img.copy()
        for idx, point in enumerate(self.points):
            cv2.circle(overlay, point, 7, (0, 255, 0), -1)
            cv2.putText(
                overlay,
                str(idx + 1),
                (point[0] + 10, point[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        if len(self.points) == 4:
            pts = np.array(self.points, dtype=np.int32)
            cv2.polylines(overlay, [pts], True, (0, 255, 255), 2)
            cv2.putText(
                overlay,
                "ENTER=confirm  S=save  BACKSPACE=reset  ESC=quit",
                (40, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                overlay,
                "Click TL, TR, BR, BL corners (right-click removes).",
                (40, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 200, 255),
                2,
                cv2.LINE_AA,
            )
        return overlay

    def select_board(
        self,
        screenshot: np.ndarray,
        origin: Tuple[int, int] = (0, 0),
    ) -> Optional[BoardSelectionResult]:
        window_title = "Select Board (click TL, TR, BR, BL)"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_title, self._mouse_callback)

        while True:
            overlay = self._draw_overlay(screenshot)
            cv2.imshow(window_title, overlay)
            key = cv2.waitKey(50) & 0xFF
            if key == 27:
                logging.info("Selection aborted by user.")
                cv2.destroyWindow(window_title)
                self.points.clear()
                return None
            if key in (8, 127):
                logging.info("Resetting board points.")
                self.points.clear()
            if key in (13, 10) and len(self.points) == 4:
                logging.info("Board selection confirmed.")
                cv2.destroyWindow(window_title)
                adjusted_points = [
                    (point[0] + origin[0], point[1] + origin[1]) for point in self.points
                ]
                return BoardSelectionResult(points=adjusted_points, save_requested=False)
            if key in (ord("s"), ord("S")) and len(self.points) == 4:
                logging.info("Board selection saved via 's' key.")
                cv2.destroyWindow(window_title)
                adjusted_points = [
                    (point[0] + origin[0], point[1] + origin[1]) for point in self.points
                ]
                return BoardSelectionResult(points=adjusted_points, save_requested=True)


class BoardReader:
    def __init__(
        self,
        transform: ScreenTransform,
        template_manager: Optional["TemplateManager"],
        debug: bool,
        yolo_detector: Optional[YOLOPieceDetector] = None,
    ) -> None:
        self.transform = transform
        self.templates = template_manager
        self.debug = debug
        self.yolo_detector = yolo_detector
        self.history: List[List[Deque[str]]] = [
            [collections.deque(maxlen=STABILITY_HISTORY) for _ in range(SQUARES_PER_SIDE)]
            for _ in range(SQUARES_PER_SIDE)
        ]
        self.mss_ctx = mss.mss()
        self.last_saved_debug_idx = 0
        if self.templates is None and self.yolo_detector is None:
            raise ValueError("BoardReader requires at least one detection backend.")

    def capture_warped(self) -> BoardDetection:
        raw = self.mss_ctx.grab(self.transform.bbox)
        raw_bgr = ensure_channel_order(np.array(raw))
        warped = self.transform.warp(raw_bgr)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        clahe_gray = create_clahe(gray)
        if self.yolo_detector is not None:
            grid = self._detect_with_yolo(warped)
        else:
            grid = self._detect_from_gray(clahe_gray)
        detection = BoardDetection(
            grid=grid,
            warped_bgr=warped,
            clahe_gray=clahe_gray,
            timestamp=time.time(),
        )
        if self.debug:
            self._show_debug_overlays(detection)
        return detection

    def _detect_with_yolo(self, warped_bgr: np.ndarray) -> List[List[SquareDetection]]:
        if self.yolo_detector is None:
            raise RuntimeError("YOLO detector not configured.")
        raw_grid = self.yolo_detector.detect(warped_bgr)
        grid: List[List[SquareDetection]] = []
        for row in range(SQUARES_PER_SIDE):
            row_detections: List[SquareDetection] = []
            for col in range(SQUARES_PER_SIDE):
                detection = raw_grid[row][col]
                stable_detection = self._stabilize_square(row, col, detection.label)
                if stable_detection is not None:
                    detection = dataclasses.replace(detection, label=stable_detection)
                row_detections.append(detection)
            grid.append(row_detections)
        return grid

    def _detect_from_gray(self, gray: np.ndarray) -> List[List[SquareDetection]]:
        if self.templates is None:
            raise RuntimeError("Template detection requested but TemplateManager not configured.")
        cell_size = BOARD_CANVAS_SIZE // SQUARES_PER_SIDE
        crop_margin = square_margin_pixels(cell_size)
        grid: List[List[SquareDetection]] = []
        for row in range(SQUARES_PER_SIDE):
            row_detections: List[SquareDetection] = []
            for col in range(SQUARES_PER_SIDE):
                y0 = row * cell_size + crop_margin
                y1 = (row + 1) * cell_size - crop_margin
                x0 = col * cell_size + crop_margin
                x1 = (col + 1) * cell_size - crop_margin
                cell_gray = gray[y0:y1, x0:x1]
                cell_edges = preprocess_edges(cell_gray)
                detection = self.templates.best_match(cell_edges)
                detection = self._apply_threshold(detection)
                stable_detection = self._stabilize_square(row, col, detection.label)
                if stable_detection is not None:
                    detection = dataclasses.replace(detection, label=stable_detection)
                row_detections.append(detection)
            grid.append(row_detections)
        return grid

    @staticmethod
    def _apply_threshold(detection: SquareDetection) -> SquareDetection:
        if detection.label == DEFAULT_PIECE_LABEL:
            return detection
        threshold = DEFAULT_WHITE_THRESHOLD if detection.label.startswith("w") else DEFAULT_BLACK_THRESHOLD
        if detection.score < threshold:
            return SquareDetection(label=DEFAULT_PIECE_LABEL, score=detection.score)
        return detection

    def _stabilize_square(self, row: int, col: int, candidate: str) -> Optional[str]:
        history = self.history[row][col]
        history.append(candidate)
        if len(history) < STABILITY_HISTORY:
            return None
        counter = collections.Counter(history)
        label, count = counter.most_common(1)[0]
        if count >= REQUIRED_STABLE_COUNT:
            return label
        return None

    def apply_move_hint(
        self,
        move: chess.Move,
        orientation: BotOrientation,
        expected_board: chess.Board,
    ) -> None:
        self._force_square_label(move.from_square, orientation, DEFAULT_PIECE_LABEL)
        piece = expected_board.piece_at(move.to_square)
        target_label = DEFAULT_PIECE_LABEL if piece is None else piece_to_detection_label(piece)
        self._force_square_label(move.to_square, orientation, target_label)

    def _force_square_label(self, square: chess.Square, orientation: BotOrientation, label: str) -> None:
        row, col = square_to_grid_indices(square, orientation)
        history = self.history[row][col]
        history.clear()
        for _ in range(STABILITY_HISTORY):
            history.append(label)

    def _show_debug_overlays(self, detection: BoardDetection) -> None:
        warped_copy = detection.warped_bgr.copy()
        cell_size = BOARD_CANVAS_SIZE // SQUARES_PER_SIDE
        for row_idx, row in enumerate(detection.grid):
            for col_idx, square in enumerate(row):
                x = col_idx * cell_size
                y = row_idx * cell_size
                cv2.rectangle(warped_copy, (x, y), (x + cell_size, y + cell_size), (50, 50, 50), 1)
                label = square.label if square.label != DEFAULT_PIECE_LABEL else "."
                cv2.putText(
                    warped_copy,
                    f"{label}:{square.score:.2f}",
                    (x + 5, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
        cv2.imshow("Warped Board", warped_copy)
        cv2.imshow("CLAHE Gray", detection.clahe_gray)
        cv2.waitKey(1)

        debug_dir = Path("debug")
        debug_dir.mkdir(exist_ok=True)
        filename = debug_dir / f"board_{self.last_saved_debug_idx:04d}.png"
        cv2.imwrite(str(filename), detection.warped_bgr)
        self.last_saved_debug_idx = (self.last_saved_debug_idx + 1) % 1000


class GameState:
    def __init__(self, desired_color: str, trust_detections: bool = False) -> None:
        self.board = chess.Board()
        self.orientation: Optional[BotOrientation] = None
        self.desired_color = desired_color
        self.our_color: Optional[chess.Color] = None
        self.move_history: List[chess.Move] = []
        self._mismatch_streak = 0
        self._initial_sync_pending = False
        self._initial_sync_attempts = 0
        self._manual_game_over = False
        self.last_large_mismatch = False
        self.last_mismatch_square_count = 0
        self.low_confidence_prompt_disabled = False
        self.trust_detections = trust_detections
        self.ambiguous_diff_streak = 0
        self.force_restart = False

    def ensure_orientation(self, detection: BoardDetection) -> None:
        if self.orientation is not None:
            return
        if self.desired_color == "white":
            orientation = BotOrientation.WHITE_BOTTOM
        elif self.desired_color == "black":
            orientation = BotOrientation.BLACK_BOTTOM
        else:
            orientation = self._infer_orientation(detection)
            if orientation is None:
                logging.debug("Orientation inference inconclusive. Waiting for better detection.")
                return

        self.orientation = orientation
        self.our_color = chess.WHITE if orientation.is_white_bottom else chess.BLACK
        self._initial_sync_pending = True
        self._initial_sync_attempts = 0
        logging.info("Orientation determined as %s. Playing %s.", orientation.value, self.side_name)

    def _infer_orientation(self, detection: BoardDetection) -> Optional[BotOrientation]:
        candidates = [BotOrientation.WHITE_BOTTOM, BotOrientation.BLACK_BOTTOM]
        scores: Dict[BotOrientation, int] = {}
        for candidate in candidates:
            piece_map = detection_grid_to_piece_map(detection, candidate)
            if not piece_map:
                scores[candidate] = 0
                continue
            white_home = sum(
                1 for square, symbol in piece_map.items() if symbol.isupper() and chess.square_rank(square) <= 3
            )
            black_home = sum(
                1 for square, symbol in piece_map.items() if symbol.islower() and chess.square_rank(square) >= 4
            )
            white_king = next((sq for sq, sym in piece_map.items() if sym == "K"), None)
            black_king = next((sq for sq, sym in piece_map.items() if sym == "k"), None)
            king_bonus = 0
            if white_king is not None and chess.square_rank(white_king) <= 3:
                king_bonus += 2
            if black_king is not None and chess.square_rank(black_king) >= 4:
                king_bonus += 2
            scores[candidate] = white_home + black_home + king_bonus

        best_orientation = max(scores, key=scores.get)
        best_score = scores[best_orientation]
        other_orientation = (
            BotOrientation.BLACK_BOTTOM if best_orientation is BotOrientation.WHITE_BOTTOM else BotOrientation.WHITE_BOTTOM
        )
        other_score = scores[other_orientation]

        if best_score == 0 or best_score == other_score:
            return None
        if best_score - other_score < 2:
            return None
        return best_orientation

    @property
    def side_name(self) -> str:
        if self.our_color is None:
            return "unknown"
        return "white" if self.our_color == chess.WHITE else "black"

    def reconcile_detection(self, piece_map: Dict[chess.Square, str]) -> Optional[chess.Move]:
        current_map = {
            square: self.board.piece_at(square).symbol()
            for square in chess.SQUARES
            if self.board.piece_at(square) is not None
        }
        self.last_large_mismatch = False
        self.last_mismatch_square_count = 0
        self.force_restart = False
        if piece_map == current_map:
            self._mismatch_streak = 0
            self.ambiguous_diff_streak = 0
            return None

        diff_squares = {sq for sq in chess.SQUARES if current_map.get(sq) != piece_map.get(sq)}
        self.last_mismatch_square_count = len(diff_squares)
        if not diff_squares:
            self._mismatch_streak = 0
            self.ambiguous_diff_streak = 0
            return None

        candidate_moves = []
        for move in self.board.legal_moves:
            temp_board = self.board.copy(stack=False)
            temp_board.push(move)
            temp_map = {
                square: temp_board.piece_at(square).symbol()
                for square in chess.SQUARES
                if temp_board.piece_at(square) is not None
            }
            if temp_map == piece_map:
                candidate_moves.append(move)

        if len(candidate_moves) == 1:
            move = candidate_moves[0]
            san = self.board.san(move)
            self.board.push(move)
            self.move_history.append(move)
            self._mismatch_streak = 0
            self.ambiguous_diff_streak = 0
            logging.info("Detected opponent move: %s (uci=%s)", san, move.uci())
            return move

        if self.trust_detections:
            if self._validate_piece_map(piece_map):
                logging.info(
                    "Trusting detection data to override board state (%d mismatches).",
                    len(diff_squares),
                )
                self._override_board(piece_map)
                if self.our_color is not None:
                    self.board.turn = self.our_color
                self._mismatch_streak = 0
                self.ambiguous_diff_streak = 0
                return None
            logging.debug(
                "Detection override skipped: missing kings or invalid piece map (%d mismatches).",
                len(diff_squares),
            )

        self._mismatch_streak += 1

        if len(diff_squares) <= 6:
            self.ambiguous_diff_streak += 1
            logging.warning(
                "Ambiguous vision diff (%d squares). Retaining previous state.",
                len(diff_squares),
            )
            if self.ambiguous_diff_streak >= AMBIGUOUS_RESTART_THRESHOLD:
                logging.warning(
                    "Ambiguous diff persisted for %d frames. Flagging restart.",
                    self.ambiguous_diff_streak,
                )
                self.force_restart = True
        else:
            logging.error("Large mismatch between detection and board state (%d squares).", len(diff_squares))
            self.last_large_mismatch = True
            self.ambiguous_diff_streak = 0

        if (
            not self.move_history
            and self._validate_piece_map(piece_map)
            and self._mismatch_streak >= 3
        ):
            logging.info("Mismatch streak hit %d with valid detection. Overriding internal board.", self._mismatch_streak)
            self._override_board(piece_map)
            self._mismatch_streak = 0
            self.ambiguous_diff_streak = 0
            self.force_restart = False
            return None
        return None

    def register_our_move(self, move: chess.Move) -> None:
        self.board.push(move)
        self.move_history.append(move)
        self._mismatch_streak = 0
        self._manual_game_over = False
        self.last_large_mismatch = False
        self.last_mismatch_square_count = 0
        self.ambiguous_diff_streak = 0
        self.force_restart = False

    def is_our_turn(self) -> bool:
        return self.our_color is not None and self.board.turn == self.our_color

    def game_over(self) -> bool:
        return self._manual_game_over or self.board.is_game_over()

    def mark_manual_game_over(self) -> None:
        self._manual_game_over = True

    @staticmethod
    def _validate_piece_map(piece_map: Dict[chess.Square, str]) -> bool:
        symbols = piece_map.values()
        return "K" in symbols and "k" in symbols

    def _override_board(self, piece_map: Dict[chess.Square, str]) -> None:
        new_board = chess.Board()
        new_board.clear_board()
        for square, symbol in piece_map.items():
            try:
                piece = chess.Piece.from_symbol(symbol)
            except ValueError:
                continue
            new_board.set_piece_at(square, piece)
        self._ensure_kings_present(new_board)
        previous_turn = self.board.turn
        had_history = bool(self.move_history)
        new_board.turn = previous_turn
        self._reset_castling_rights(new_board)
        new_board.halfmove_clock = 0
        new_board.fullmove_number = max(1, len(self.move_history) // 2 + 1)
        self.board = new_board
        self.move_history.clear()
        self._mismatch_streak = 0
        self._initial_sync_pending = False
        self._initial_sync_attempts = 0
        self._manual_game_over = False
        if not had_history and self.our_color is not None:
            self.board.turn = self.our_color
        self.last_large_mismatch = False
        self.last_mismatch_square_count = 0

    def _ensure_kings_present(self, board: chess.Board) -> None:
        for color in (chess.WHITE, chess.BLACK):
            if board.king(color) is None:
                fallback = self.board.king(color)
                if fallback is not None:
                    board.set_piece_at(fallback, chess.Piece(chess.KING, color))
                else:
                    # Place king at default starting square if fallback unavailable
                    default_square = chess.E1 if color == chess.WHITE else chess.E8
                    board.set_piece_at(default_square, chess.Piece(chess.KING, color))

    @staticmethod
    def _reset_castling_rights(board: chess.Board) -> None:
        if hasattr(board, "clear_castling_rights"):
            board.clear_castling_rights()  # type: ignore[attr-defined]
        elif hasattr(board, "clean_castling_rights"):
            board.clean_castling_rights()  # type: ignore[attr-defined]
        else:
            board.castling_rights = 0  # type: ignore[attr-defined]

    def maybe_apply_initial_sync(self, piece_map: Dict[chess.Square, str]) -> None:
        if not self._initial_sync_pending:
            return
        self._initial_sync_attempts += 1
        if not piece_map:
            logging.debug("Initial sync pending but detection map empty; waiting.")
            return
        if self._validate_piece_map(piece_map):
            logging.info("Initial board state synchronized from detection.")
            self._override_board(piece_map)
            return
        if self._initial_sync_attempts >= 5:
            logging.warning(
                "Initial sync pending but kings not detected after %d attempts. Overriding with available pieces.",
                self._initial_sync_attempts,
            )
            self._override_board(piece_map)


class EngineWrapper:
    def __init__(self, config: CLIConfig) -> None:
        engine_path = config.engine_path or os.environ.get("STOCKFISH_BINARY")
        if engine_path:
            engine_path = os.path.expanduser(engine_path)
            if not Path(engine_path).exists():
                raise FileNotFoundError(f"Stockfish binary not found at {engine_path}")
        else:
            engine_path = "stockfish"

        logging.info("Initializing Stockfish engine (%s)", engine_path)
        self.config = config
        self.engine_path = engine_path
        self.skill_level = 20
        self.engine = self._create_engine()

    def _create_engine(self) -> Stockfish:
        engine = Stockfish(path=self.engine_path)
        self._configure_engine(engine)
        return engine

    def _configure_engine(self, engine: Stockfish) -> None:
        self._apply_skill_level(engine, log_context="initial setup")
        engine.set_depth(self.config.depth)
        if not self.config.move_time_ms:
            self.config.move_time_ms = 3000

    def _choose_skill_level(self) -> int:
        power_range = self.config.stockfish_power_range
        if power_range is not None:
            low, high = power_range
            return random.randint(low, high)
        return 20

    def _apply_skill_level(self, engine: Stockfish, log_context: Optional[str] = None) -> None:
        skill_level = self._choose_skill_level()
        engine.set_skill_level(skill_level)
        self.skill_level = skill_level
        if self.config.stockfish_power_range is not None:
            if log_context:
                logging.info("Stockfish skill set to %d (%s).", skill_level, log_context)
            else:
                logging.info("Stockfish skill set to %d.", skill_level)

    def best_move(self, board: chess.Board) -> Optional[chess.Move]:
        move_str: Optional[str] = None
        for attempt in range(2):
            self._apply_skill_level(self.engine, log_context="current move")
            try:
                self.engine.set_fen_position(board.fen())
                if self.config.move_time_ms:
                    move_str = self.engine.get_best_move_time(self.config.move_time_ms)
                else:
                    move_str = self.engine.get_best_move()
                break
            except StockfishException as exc:  # pragma: no cover - depends on external engine
                logging.error("Stockfish crashed while retrieving best move (%s). Restarting engine.", exc)
                self.engine = self._create_engine()
                if attempt == 1:
                    logging.error("Stockfish failed again after restart. Skipping move.")
                    return None
        if move_str is None:
            return None
        try:
            move = chess.Move.from_uci(move_str)
        except ValueError as exc:
            logging.error("Engine returned invalid move %s (%s)", move_str, exc)
            return None
        if move not in board.legal_moves:
            logging.warning("Engine move %s not legal in current position.", move_str)
            return None
        return move

    def close(self) -> None:
        try:
            quit_method = getattr(self.engine, "quit", None)
            if callable(quit_method):
                quit_method()
            else:
                write_command = getattr(self.engine, "write_command", None)
                if callable(write_command):
                    write_command("quit")
        except AttributeError:
            # Older stockfish package versions expose neither quit nor write_command.
            pass
        except Exception:
            logging.debug("Error while shutting down Stockfish engine.", exc_info=True)


def square_center_on_canvas(square: chess.Square, orientation: BotOrientation) -> Tuple[float, float]:
    file_idx = chess.square_file(square)
    rank_idx = chess.square_rank(square)
    if orientation.is_white_bottom:
        col = file_idx
        row = 7 - rank_idx
    else:
        col = 7 - file_idx
        row = rank_idx
    cell_size = BOARD_CANVAS_SIZE / SQUARES_PER_SIDE
    center_x = (col + 0.5) * cell_size
    center_y = (row + 0.5) * cell_size
    return center_x, center_y


@dataclasses.dataclass(frozen=True)
class MovePreviewHandle:
    token: int


class ScreenArrowDrawer:
    PS_SOLID = 0
    R2_XORPEN = 7

    def __init__(self) -> None:
        if os.name != "nt":
            raise RuntimeError("Move preview overlay requires Windows APIs.")
        self.user32 = ctypes.windll.user32
        self.gdi32 = ctypes.windll.gdi32
        self.lock = threading.Lock()
        self._last_points: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
        self.pen_width = 4
        self.color = 0x00FFA500

    def draw(self, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        with self.lock:
            if self._last_points is not None:
                self._xor_arrow(*self._last_points)
                self._last_points = None
            if not self._xor_arrow(start, end):
                return False
            self._last_points = (start, end)
            return True

    def clear(self) -> None:
        with self.lock:
            if self._last_points is not None:
                self._xor_arrow(*self._last_points)
                self._last_points = None

    def _xor_arrow(self, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        if start == end:
            return False
        hdc = self.user32.GetDC(0)
        if not hdc:
            return False
        try:
            self.gdi32.SetROP2(hdc, self.R2_XORPEN)
            pen = self.gdi32.CreatePen(self.PS_SOLID, self.pen_width, self.color)
            if not pen:
                return False
            old_pen = self.gdi32.SelectObject(hdc, pen)
            self._line(hdc, start, end)
            arrow_points = self._arrowhead_points(start, end)
            if arrow_points is not None:
                left, right = arrow_points
                self._line(hdc, end, left)
                self._line(hdc, end, right)
            self.gdi32.SelectObject(hdc, old_pen)
            self.gdi32.DeleteObject(pen)
            return True
        finally:
            self.user32.ReleaseDC(0, hdc)

    def _line(self, hdc: int, start: Tuple[int, int], end: Tuple[int, int]) -> None:
        self.gdi32.MoveToEx(hdc, int(start[0]), int(start[1]), None)
        self.gdi32.LineTo(hdc, int(end[0]), int(end[1]))

    @staticmethod
    def _arrowhead_points(
        start: Tuple[int, int],
        end: Tuple[int, int],
    ) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length < 8.0:
            return None
        arrow_len = min(max(int(length * 0.2), 16), 60)
        base_angle = math.atan2(dy, dx)
        spread = math.radians(28.0)
        left_angle = base_angle - spread
        right_angle = base_angle + spread
        left = (
            int(round(end[0] - arrow_len * math.cos(left_angle))),
            int(round(end[1] - arrow_len * math.sin(left_angle))),
        )
        right = (
            int(round(end[0] - arrow_len * math.cos(right_angle))),
            int(round(end[1] - arrow_len * math.sin(right_angle))),
        )
        return left, right


class MovePreviewRenderer:
    def __init__(self, transform: ScreenTransform) -> None:
        self.transform = transform
        self._drawer: Optional[ScreenArrowDrawer] = None
        try:
            self._drawer = ScreenArrowDrawer()
        except Exception:
            logging.warning(
                "Move preview overlay disabled; failed to initialise screen arrow renderer.",
                exc_info=True,
            )

    def show(self, move: chess.Move, orientation: BotOrientation) -> bool:
        if self._drawer is None:
            return False
        points = self._move_points(move, orientation)
        if points is None:
            return False
        start_pt, end_pt = points
        return self._drawer.draw(start_pt, end_pt)

    def hide(self) -> None:
        if self._drawer is not None:
            self._drawer.clear()

    def _move_points(
        self, move: chess.Move, orientation: BotOrientation
    ) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        src_center = square_center_on_canvas(move.from_square, orientation)
        dst_center = square_center_on_canvas(move.to_square, orientation)
        src_x, src_y = self.transform.normalized_center_to_screen(src_center)
        dst_x, dst_y = self.transform.normalized_center_to_screen(dst_center)
        start = self._clamp_point((int(round(src_x)), int(round(src_y))))
        end = self._clamp_point((int(round(dst_x)), int(round(dst_y))))
        if start is None or end is None:
            return None
        return start, end

    def _clamp_point(self, point: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        bbox = self.transform.bbox
        width = int(bbox["width"])
        height = int(bbox["height"])
        if width <= 0 or height <= 0:
            return None
        x, y = point
        left = int(bbox["left"])
        top = int(bbox["top"])
        right = left + width - 1
        bottom = top + height - 1
        x = max(left, min(right, x))
        y = max(top, min(bottom, y))
        return (x, y)


class MouseController:
    def __init__(self, transform: ScreenTransform) -> None:
        self.transform = transform
        self.preview_renderer = MovePreviewRenderer(transform)
        self._preview_counter = 0
        self._active_preview: Optional[MovePreviewHandle] = None
        pyautogui.FAILSAFE = True

    @staticmethod
    def _add_jitter(x: float, y: float, magnitude: float = 3.0) -> Tuple[float, float]:
        return x + random.uniform(-magnitude, magnitude), y + random.uniform(-magnitude, magnitude)

    def _square_to_center(self, square: chess.Square, orientation: BotOrientation) -> Tuple[float, float]:
        return square_center_on_canvas(square, orientation)

    def _centers_to_screen(
        self,
        move: chess.Move,
        orientation: BotOrientation,
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        src_center = self._square_to_center(move.from_square, orientation)
        dst_center = self._square_to_center(move.to_square, orientation)
        src_x, src_y = self.transform.normalized_center_to_screen(src_center)
        dst_x, dst_y = self.transform.normalized_center_to_screen(dst_center)
        return (src_x, src_y), (dst_x, dst_y)

    def perform_drag(self, move: chess.Move, orientation: BotOrientation) -> None:
        (src_x, src_y), (dst_x, dst_y) = self._centers_to_screen(move, orientation)
        src_x, src_y = self._add_jitter(src_x, src_y)
        dst_x, dst_y = self._add_jitter(dst_x, dst_y)
        pyautogui.moveTo(src_x, src_y, duration=random.uniform(0.12, 0.25))
        pyautogui.mouseDown()
        time.sleep(random.uniform(0.05, 0.12))
        pyautogui.moveTo(dst_x, dst_y, duration=random.uniform(0.20, 0.35))
        time.sleep(random.uniform(0.04, 0.09))
        pyautogui.mouseUp()

    def perform_click(self, move: chess.Move, orientation: BotOrientation) -> None:
        (src_x, src_y), (dst_x, dst_y) = self._centers_to_screen(move, orientation)
        src_x, src_y = self._add_jitter(src_x, src_y)
        dst_x, dst_y = self._add_jitter(dst_x, dst_y)
        pyautogui.moveTo(src_x, src_y, duration=random.uniform(0.08, 0.18))
        pyautogui.click()
        time.sleep(random.uniform(0.05, 0.10))
        pyautogui.moveTo(dst_x, dst_y, duration=random.uniform(0.08, 0.18))
        pyautogui.click()

    def click_square(
        self,
        square: chess.Square,
        orientation: BotOrientation,
        delay: float = 0.0,
    ) -> None:
        if delay > 0:
            time.sleep(delay)
        center_x, center_y = self._square_to_center(square, orientation)
        center_x, center_y = self._add_jitter(center_x, center_y, magnitude=2.0)
        pyautogui.moveTo(center_x, center_y, duration=random.uniform(0.05, 0.12))
        pyautogui.click()

    def begin_move_preview(
        self,
        move: chess.Move,
        orientation: BotOrientation,
    ) -> Optional[MovePreviewHandle]:
        if self._active_preview is not None:
            self.preview_renderer.hide()
            self._active_preview = None
        success = self.preview_renderer.show(move, orientation)
        if not success:
            return None
        self._preview_counter += 1
        handle = MovePreviewHandle(token=self._preview_counter)
        self._active_preview = handle
        return handle

    def end_move_preview(self, handle: Optional[MovePreviewHandle]) -> None:
        if handle is None:
            return
        if self._active_preview != handle:
            return
        self.preview_renderer.hide()
        self._active_preview = None

    def handle_promotion(self, orientation: BotOrientation, promotion: str) -> None:
        logging.info("Awaiting promotion UI for piece '%s'.", promotion)
        time.sleep(0.6)
        promotion = promotion.lower()
        rank_offset = -1 if orientation.is_white_bottom else 1
        pyautogui.moveRel(0, rank_offset * 60, duration=0.1)
        index_map = {"q": 0, "r": 1, "b": 2, "n": 3}
        idx = index_map.get(promotion, 0)
        pyautogui.moveRel(idx * 60, 0, duration=0.1)
        pyautogui.click()


def take_full_screenshot() -> Tuple[np.ndarray, Tuple[int, int]]:
    with mss.mss() as capture:
        monitors = capture.monitors
        monitor_to_use = monitors[0]
        mouse_x, mouse_y = pyautogui.position()
        for monitor in monitors[1:]:
            left = monitor["left"]
            top = monitor["top"]
            width = monitor["width"]
            height = monitor["height"]
            if left <= mouse_x < left + width and top <= mouse_y < top + height:
                monitor_to_use = monitor
                break
        screen = capture.grab(monitor_to_use)
    origin = (monitor_to_use["left"], monitor_to_use["top"])
    return ensure_channel_order(np.array(screen)), origin


class BoardCoordinateCache:
    def __init__(self, path: Path = BOARD_COORDINATES_FILE) -> None:
        self.path = path
        self.coords: Dict[str, List[List[int]]] = {}
        self.last_used: Optional[str] = None
        self._load()

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text())
        except FileNotFoundError:
            return
        except Exception:
            logging.warning("Failed to read board coordinate cache %s.", self.path, exc_info=True)
            return

        coords_data: Optional[Dict[str, List[List[int]]]] = None
        last_used: Optional[str] = None

        if isinstance(raw, dict):
            potential_coords = raw.get("coords")
            if isinstance(potential_coords, dict):
                coords_data = self._normalize_coords_dict(potential_coords)
            elif self._looks_like_coords_dict(raw):
                coords_data = self._normalize_coords_dict(raw)
            last_val = raw.get("last_used")
            if isinstance(last_val, str):
                last_used = last_val
        elif isinstance(raw, list):
            normalized = self._normalize_points(raw)
            if normalized is not None:
                coords_data = {"1": normalized}
                last_used = "1"

        if coords_data is not None:
            self.coords = coords_data
            if last_used and last_used in self.coords:
                self.last_used = last_used
            elif self.coords:
                self.last_used = next(iter(sorted(self.coords.keys())))
        else:
            logging.warning("Board coordinate cache %s has unexpected structure; ignoring.", self.path)

    def _normalize_coords_dict(self, source: Dict[str, object]) -> Optional[Dict[str, List[List[int]]]]:
        normalized: Dict[str, List[List[int]]] = {}
        for key, value in source.items():
            if key == "last_used":
                continue
            points = self._normalize_points(value)
            if points is None:
                logging.warning("Skipping invalid coordinate entry '%s' in %s.", key, self.path)
                continue
            normalized[key] = points
        if not normalized:
            return None
        return normalized

    @staticmethod
    def _looks_like_coords_dict(raw: Dict[str, object]) -> bool:
        return all(isinstance(k, str) for k in raw.keys())

    @staticmethod
    def _normalize_points(value: object) -> Optional[List[List[int]]]:
        if not isinstance(value, (list, tuple)):
            return None
        points: List[List[int]] = []
        for entry in value:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                return None
            try:
                x = int(entry[0])
                y = int(entry[1])
            except (TypeError, ValueError):
                return None
            points.append([x, y])
        if len(points) != 4:
            return None
        return points

    def available_ids(self) -> List[str]:
        def sort_key(value: str) -> Tuple[int, Union[int, str]]:
            return (0, int(value)) if value.isdigit() else (1, value)

        return sorted(self.coords.keys(), key=sort_key)

    def get(self, coord_id: Optional[Union[str, int]]) -> Optional[Tuple[List[Tuple[int, int]], str]]:
        target_id: Optional[str]
        if coord_id is None:
            target_id = self.last_used
        else:
            target_id = str(coord_id)
        if target_id is None:
            return None
        raw_points = self.coords.get(target_id)
        if raw_points is None:
            return None
        points = [(int(x), int(y)) for x, y in raw_points]
        return points, target_id

    def save(
        self,
        points: Sequence[Tuple[int, int]],
        coord_id: Optional[Union[str, int]] = None,
        mark_last: bool = True,
    ) -> str:
        normalized = self._normalize_points(list(points))
        if normalized is None:
            raise ValueError("Board coordinates must contain exactly four (x, y) pairs.")
        if coord_id is None:
            coord_id = self._allocate_id()
        target_id = str(coord_id)
        self.coords[target_id] = normalized
        if mark_last:
            self.last_used = target_id
        self._write()
        logging.info("Saved board coordinates under id %s to %s.", target_id, self.path)
        return target_id

    def mark_last_used(self, coord_id: Union[str, int]) -> None:
        target_id = str(coord_id)
        if target_id not in self.coords:
            return
        if self.last_used == target_id:
            return
        self.last_used = target_id
        self._write()

    def delete(self, coord_id: Union[str, int]) -> bool:
        target_id = str(coord_id)
        if target_id not in self.coords:
            return False
        del self.coords[target_id]
        if self.last_used == target_id:
            remaining = self.available_ids()
            self.last_used = remaining[0] if remaining else None
        self._write()
        logging.info("Removed cached board coordinates id %s from %s.", target_id, self.path)
        return True

    def _allocate_id(self) -> str:
        numeric_ids = [
            int(key)
            for key in self.coords.keys()
            if isinstance(key, str) and key.isdigit()
        ]
        next_id = max(numeric_ids, default=0) + 1
        return str(next_id)

    def _write(self) -> None:
        data = {
            "coords": self.coords,
            "last_used": self.last_used,
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(data, indent=2))
        except Exception:
            logging.warning("Failed to write board coordinate cache %s.", self.path, exc_info=True)


def compute_transform(points: List[Tuple[int, int]]) -> ScreenTransform:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    left = min(xs)
    right = max(xs)
    top = min(ys)
    bottom = max(ys)
    bbox = {
        "left": left,
        "top": top,
        "width": right - left,
        "height": bottom - top,
    }
    src_pts = np.array(
        [
            [points[0][0] - left, points[0][1] - top],
            [points[1][0] - left, points[1][1] - top],
            [points[2][0] - left, points[2][1] - top],
            [points[3][0] - left, points[3][1] - top],
        ],
        dtype=np.float32,
    )
    dst_pts = np.array(
        [
            [0, 0],
            [BOARD_CANVAS_SIZE - 1, 0],
            [BOARD_CANVAS_SIZE - 1, BOARD_CANVAS_SIZE - 1],
            [0, BOARD_CANVAS_SIZE - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    matrix_inv = np.linalg.inv(matrix)
    return ScreenTransform(
        bbox=bbox,
        src_points=src_pts,
        dst_points=dst_pts,
        matrix=matrix,
        matrix_inv=matrix_inv,
        canvas_size=(BOARD_CANVAS_SIZE, BOARD_CANVAS_SIZE),
    )


def wait_for_opponent(
    reader: BoardReader,
    state: GameState,
    orientation: BotOrientation,
    confidence_floor: float,
    timeout: float = 60.0,
    board_points: Optional[Sequence[Tuple[int, int]]] = None,
    prompt_threshold: int = ALERT_PROMPT_THRESHOLD,
    restart_threshold: int = ALERT_RESTART_THRESHOLD,
    stop_event: Optional[threading.Event] = None,
    key_provider: Optional[Callable[[], Optional[str]]] = None,
) -> None:
    if state.game_over():
        logging.info("Game already finished. Skipping opponent wait.")
        return

    if key_provider is None:
        key_provider = console_key_pressed

    def stop_requested() -> bool:
        return stop_event is not None and stop_event.is_set()

    start = time.time()
    low_confidence_hits = 0
    prompt_shown = False
    alert_disabled = state.low_confidence_prompt_disabled
    large_mismatch_hits = 0
    mismatch_prompt_shown = False
    while time.time() - start < timeout:
        if stop_requested():
            logging.info("Stop requested while waiting for opponent.")
            raise KeyboardInterrupt
        if state.game_over():
            logging.info("Detected game end while waiting for opponent.")
            return
        detection = reader.capture_warped()
        occupied_scores = [
            sq.score for row in detection.grid for sq in row if sq.label != DEFAULT_PIECE_LABEL
        ]
        if occupied_scores and min(occupied_scores) < confidence_floor:
            logging.debug("Opponent wait: detection confidence low (%.2f).", min(occupied_scores))
            low_confidence_hits += 1
            if (
                not alert_disabled
                and not prompt_shown
                and low_confidence_hits >= prompt_threshold
            ):
                response = prompt_match_end_confirmation(low_confidence_hits)
                prompt_shown = True
                if response is True:
                    logging.info("User confirmed the match ended while waiting for opponent.")
                    state.mark_manual_game_over()
                    return
                if response is False:
                    logging.info("Continuing to wait for opponent as per user request.")
                    alert_disabled = True
                    state.low_confidence_prompt_disabled = True
            if low_confidence_hits >= restart_threshold:
                logging.warning(
                    "Low-confidence detection persisted for %d checks. Restarting session.",
                    low_confidence_hits,
                )
                if board_points:
                    raise RestartRequested(board_points)
                logging.warning("Board coordinates unavailable; unable to restart automatically.")
                low_confidence_hits = 0
                prompt_shown = False
            time.sleep(0.4)
            continue
        else:
            low_confidence_hits = 0
            prompt_shown = False
            if not state.low_confidence_prompt_disabled:
                alert_disabled = False
        piece_map = detection_grid_to_piece_map(detection, orientation)
        move = state.reconcile_detection(piece_map)
        if state.force_restart:
            logging.warning(
                "Ambiguous diff persisted for %d frames while waiting. Restarting session.",
                state.ambiguous_diff_streak,
            )
            if board_points:
                raise RestartRequested(board_points)
            logging.warning("Board coordinates unavailable; unable to restart automatically.")
            state.ambiguous_diff_streak = 0
            continue
        if move is not None:
            return
        if state.last_large_mismatch:
            large_mismatch_hits += 1
            if (
                not alert_disabled
                and not mismatch_prompt_shown
                and large_mismatch_hits >= prompt_threshold
            ):
                response = prompt_match_end_confirmation(large_mismatch_hits)
                mismatch_prompt_shown = True
                if response is True:
                    logging.info("User confirmed the match ended after repeated mismatches.")
                    state.mark_manual_game_over()
                    return
                if response is False:
                    logging.info("Continuing despite mismatches as per user request.")
                    alert_disabled = True
                    state.low_confidence_prompt_disabled = True
            if large_mismatch_hits >= restart_threshold:
                logging.warning(
                    "Large board mismatch persisted for %d checks. Restarting session.",
                    large_mismatch_hits,
                )
                if board_points:
                    raise RestartRequested(board_points)
                logging.warning("Board coordinates unavailable; unable to restart automatically.")
                large_mismatch_hits = 0
                mismatch_prompt_shown = False
        else:
            large_mismatch_hits = 0
            mismatch_prompt_shown = False
            if not state.low_confidence_prompt_disabled:
                alert_disabled = False
        if stop_requested():
            logging.info("Stop requested while waiting for opponent.")
            raise KeyboardInterrupt
        key = key_provider()
        if key == "q":
            logging.info("Quit requested while waiting for opponent.")
            raise KeyboardInterrupt
        if key == "r":
            logging.info("Reselection requested while waiting (deferred).")
        time.sleep(0.4)
    logging.warning("Timeout while waiting for opponent move.")


def run_bot(config: CLIConfig, stop_event: Optional[threading.Event] = None) -> None:
    logging.info("Booting Screen Chess Bot %s – hold tight while I warm up Stockfish.", __version__)
    cache = BoardCoordinateCache()
    cached_points: Optional[List[Tuple[int, int]]] = None
    active_coord_id: Optional[str] = None
    restart_coord_id: Optional[str] = None

    def stop_requested() -> bool:
        return stop_event is not None and stop_event.is_set()

    def poll_key() -> Optional[str]:
        if stop_requested():
            return "q"
        return console_key_pressed()

    def wait_for_manual_execution(
        expected_map: Dict[chess.Square, str],
        critical: Set[chess.Square],
        timeout: float = 25.0,
        check_interval: float = 0.8,
    ) -> bool:
        end_time = time.time() + timeout
        while time.time() < end_time:
            if stop_requested():
                return False
            if verify_expected_position(
                reader,
                state.orientation,
                expected_map,
                attempts=1,
                delay=0.25,
                critical_squares=critical,
            ):
                return True
            time.sleep(check_interval)
        return False

    if config.use_cached_coord is not None:
        loaded = cache.get(config.use_cached_coord)
        if loaded is None:
            logging.warning(
                "Requested cached coordinate id %s not found in %s.",
                config.use_cached_coord,
                cache.path,
            )
        else:
            cached_points, active_coord_id = loaded
            logging.info(
                "Loaded cached board coordinates id %s from %s.",
                active_coord_id,
                cache.path,
            )
            cache.mark_last_used(active_coord_id)

    while True:
        if stop_requested():
            logging.info("Stop requested before board selection. Exiting.")
            break
        selector = ScreenSelector()
        selection_save_requested = False
        if cached_points is None:
            full_screen, origin = take_full_screenshot()
            selection = selector.select_board(full_screen, origin)
            if selection is None:
                logging.info("No board selected. Exiting.")
                return
            current_points = list(selection.points)
            selection_save_requested = selection.save_requested
        else:
            if active_coord_id is not None:
                logging.info("Reusing stored board coordinates id %s.", active_coord_id)
            else:
                logging.info("Reusing stored board coordinates.")
            current_points = list(cached_points)

        cached_points = list(current_points)
        if selection_save_requested:
            save_id_hint: Optional[str] = active_coord_id if active_coord_id is not None else None
            try:
                active_coord_id = cache.save(
                    cached_points,
                    coord_id=save_id_hint,
                    mark_last=True,
                )
            except ValueError as exc:
                logging.warning("Unable to save board coordinates: %s", exc)
            else:
                logging.info("Using cached coordinate id %s for current session.", active_coord_id)
        elif active_coord_id is not None:
            cache.mark_last_used(active_coord_id)

        transform = compute_transform(current_points)
        yolo_weights = config.yolo_weights or os.environ.get("YOLO_WEIGHTS")
        if not yolo_weights:
            logging.error("YOLO weights path not provided. Use --yolo-weights or set YOLO_WEIGHTS.")
            return

        try:
            yolo_detector = YOLOPieceDetector(
                weights_path=Path(yolo_weights),
                confidence=config.yolo_confidence,
                iou=config.yolo_iou,
                device=config.yolo_device,
            )
        except Exception:
            logging.exception("Failed to initialize YOLO detector.")
            return

        confidence_floor = max(config.yolo_confidence - 0.05, 0.1)
        template_manager: Optional[TemplateManager] = None
        reader = BoardReader(transform, template_manager, debug=config.show_debug, yolo_detector=yolo_detector)
        state = GameState(config.color, config.trust_detections)
        engine = EngineWrapper(config)
        mouse = MouseController(transform)

        logging.warning("This bot will move your mouse. Keep the game visible. Press 'q' in console to stop.")
        if config.single_move_hotkeys:
            logging.info("Hotkeys: press 'd' to preview one move, 's' to play one move automatically.")

        first_move = True
        restart_points: Optional[List[Tuple[int, int]]] = None
        restart_coord_id = active_coord_id
        low_confidence_hits = 0
        low_conf_prompt_shown = False
        large_mismatch_hits = 0
        large_mismatch_prompt_shown = False
        alert_disabled = state.low_confidence_prompt_disabled
        prompt_threshold = ALERT_PROMPT_THRESHOLD
        restart_threshold = ALERT_RESTART_THRESHOLD
        try:
            while True:
                if stop_requested():
                    logging.info("Stop requested during gameplay loop.")
                    break
                detection = reader.capture_warped()
                state.ensure_orientation(detection)
                if state.orientation is None:
                    logging.debug("Orientation not confirmed yet. Waiting for better detection.")
                    time.sleep(0.5)
                    continue

                if state.game_over():
                    logging.info("Game concluded (%s). Exiting.", state.board.result())
                    break

                hotkey_trigger: Optional[str] = None
                if config.single_move_hotkeys:
                    hotkey_trigger = poll_key()
                    if hotkey_trigger == "q":
                        logging.info("Quit requested via hotkey. Stopping bot.")
                        break

                piece_map = detection_grid_to_piece_map(detection, state.orientation)
                state.maybe_apply_initial_sync(piece_map)
                move = state.reconcile_detection(piece_map)
                if state.force_restart:
                    logging.warning(
                        "Ambiguous diff persisted for %d frames. Restarting session.",
                        state.ambiguous_diff_streak,
                    )
                    restart_points = list(current_points)
                    restart_coord_id = active_coord_id
                    break

                occupied_scores = [
                    sq.score for row in detection.grid for sq in row if sq.label != DEFAULT_PIECE_LABEL
                ]
                if occupied_scores and min(occupied_scores) < confidence_floor:
                    logging.warning(
                        "Low detection confidence (min score %.2f). Waiting for stability.",
                        min(occupied_scores),
                    )
                    low_confidence_hits += 1
                    if (
                        not alert_disabled
                        and not low_conf_prompt_shown
                        and low_confidence_hits >= prompt_threshold
                    ):
                        response = prompt_match_end_confirmation(low_confidence_hits)
                        low_conf_prompt_shown = True
                        if response is True:
                            logging.info("User confirmed the match ended after repeated low-confidence frames.")
                            state.mark_manual_game_over()
                            break
                        if response is False:
                            logging.info("Continuing to wait despite low confidence as per user request.")
                            alert_disabled = True
                            state.low_confidence_prompt_disabled = True
                    if low_confidence_hits >= restart_threshold:
                        logging.warning(
                            "Low-confidence detection persisted for %d checks. Restarting session.",
                            low_confidence_hits,
                        )
                        restart_points = list(current_points)
                        restart_coord_id = active_coord_id
                        break
                    time.sleep(0.5)
                    continue
                else:
                    low_confidence_hits = 0
                    low_conf_prompt_shown = False
                    if not state.low_confidence_prompt_disabled:
                        alert_disabled = False

                if stop_requested():
                    logging.info("Stop requested by UI.")
                    break
                key = hotkey_trigger if config.single_move_hotkeys else poll_key()
                if key == "q":
                    logging.info("Quit requested by user.")
                    break
                if key == "r":
                    logging.info("Reselecting board as requested.")
                    full_screen, origin = take_full_screenshot()
                    new_selection = selector.select_board(full_screen, origin)
                    if new_selection is None:
                        logging.info("Reselection aborted. Continuing with previous transform.")
                    else:
                        current_points = list(new_selection.points)
                        cached_points = list(current_points)
                        if new_selection.save_requested:
                            save_id_hint = active_coord_id if active_coord_id is not None else None
                            try:
                                active_coord_id = cache.save(
                                    cached_points,
                                    coord_id=save_id_hint,
                                    mark_last=True,
                                )
                            except ValueError as exc:
                                logging.warning("Unable to save board coordinates: %s", exc)
                            else:
                                logging.info(
                                    "Updated cached coordinate id %s after reselection.",
                                    active_coord_id,
                                )
                        elif active_coord_id is not None:
                            cache.mark_last_used(active_coord_id)
                        transform = compute_transform(current_points)
                        reader = BoardReader(
                            transform,
                            template_manager,
                            debug=config.show_debug,
                            yolo_detector=yolo_detector,
                        )
                        state = GameState(config.color, config.trust_detections)
                        mouse = MouseController(transform)
                        logging.info("Transform updated. Resetting state.")
                        low_confidence_hits = 0
                        low_conf_prompt_shown = False
                        large_mismatch_hits = 0
                        large_mismatch_prompt_shown = False
                        alert_disabled = state.low_confidence_prompt_disabled
                        continue

                if not state.is_our_turn():
                    if config.single_move_hotkeys and key in ("d", "s"):
                        logging.info("Hotkey '%s' pressed but it is not our turn yet.", key)
                    time.sleep(0.2)
                    continue

                if config.single_move_hotkeys and hotkey_trigger not in ("d", "s"):
                    time.sleep(0.1)
                    continue

                move = engine.best_move(state.board)
                if move is None:
                    logging.error("Engine failed to produce a move. Aborting.")
                    break

                logging.info("Engine move: %s (uci=%s)", state.board.san(move), move.uci())
                preview_handle: Optional[MovePreviewHandle] = None
                manual_preview_mode = config.show_move_only and not (
                    config.single_move_hotkeys and hotkey_trigger == "s"
                )

                if config.single_move_hotkeys and hotkey_trigger == "d":
                    logging.info("Hotkey 'd' pressed. Showing suggested move %s.", move.uci())
                    preview_handle = mouse.begin_move_preview(move, state.orientation)
                    if preview_handle is not None:
                        time.sleep(5.0)
                        mouse.end_move_preview(preview_handle)
                    else:
                        logging.warning("Unable to draw preview arrow for suggested move.")
                    first_move = False
                    continue

                if not config.single_move_hotkeys and first_move:
                    for remaining in range(3, 0, -1):
                        logging.info("Executing move in %d...", remaining)
                        time.sleep(1)
                    first_move = False

                expected_board = state.board.copy(stack=False)
                expected_board.push(move)
                expected_map = board_piece_map(expected_board)

                promotion_choice = PROMOTION_MAP.get(move.promotion, config.promotion)
                critical_squares: Set[chess.Square] = {move.from_square, move.to_square}

                if (
                    config.move_delay_range_s
                    and not manual_preview_mode
                    and not (config.single_move_hotkeys and hotkey_trigger == "s")
                ):
                    delay = random.uniform(*config.move_delay_range_s)
                    logging.info("Delaying move execution by %.2f seconds.", delay)
                    time.sleep(delay)

                preview_handle = mouse.begin_move_preview(move, state.orientation)

                if manual_preview_mode:
                    if preview_handle is not None:
                        time.sleep(5.0)
                        mouse.end_move_preview(preview_handle)
                        preview_handle = None
                else:
                    if preview_handle is not None:
                        time.sleep(0.25)
                        mouse.end_move_preview(preview_handle)
                        preview_handle = None

                if manual_preview_mode:
                    logging.info("Preview mode active; waiting for you to play %s.", move.uci())
                    if not wait_for_manual_execution(expected_map, critical_squares):
                        logging.warning("Move %s was not detected in time; re-evaluating.", move.uci())
                        continue
                else:
                    mouse.perform_drag(move, state.orientation)
                    if move.promotion is not None:
                        mouse.click_square(move.to_square, state.orientation, delay=1.0)
                        mouse.handle_promotion(state.orientation, promotion_choice)

                    time.sleep(0.6)
                    move_confirmed = verify_expected_position(
                        reader,
                        state.orientation,
                        expected_map,
                        critical_squares=critical_squares,
                    )
                    if not move_confirmed:
                        logging.info("Drag attempt did not match expected board; trying click-to-move fallback.")
                        mouse.perform_click(move, state.orientation)
                        if move.promotion is not None:
                            mouse.click_square(move.to_square, state.orientation, delay=1.0)
                            mouse.handle_promotion(state.orientation, promotion_choice)
                        time.sleep(0.6)
                        move_confirmed = verify_expected_position(
                            reader,
                            state.orientation,
                            expected_map,
                            critical_squares=critical_squares,
                        )
                        if not move_confirmed:
                            logging.error("Unable to confirm move execution on screen. Aborting turn.")
                            continue

                reader.apply_move_hint(move, state.orientation, expected_board)
                state.register_our_move(move)
                first_move = False

                if config.single_move_hotkeys and hotkey_trigger == "s":
                    logging.info("Hotkey 's' move played. Awaiting further commands.")
                    continue

                try:
                    wait_for_opponent(
                        reader,
                        state,
                        state.orientation,
                        confidence_floor,
                        board_points=current_points,
                        stop_event=stop_event,
                        key_provider=poll_key,
                    )
                except RestartRequested as restart_exc:
                    restart_points = list(restart_exc.points)
                    restart_coord_id = active_coord_id
                    break

                if state.last_large_mismatch:
                    large_mismatch_hits += 1
                    if (
                        not alert_disabled
                        and not large_mismatch_prompt_shown
                        and large_mismatch_hits >= prompt_threshold
                    ):
                        response = prompt_match_end_confirmation(large_mismatch_hits)
                        large_mismatch_prompt_shown = True
                        if response is True:
                            logging.info("User confirmed the match ended after repeated mismatches.")
                            state.mark_manual_game_over()
                            break
                        if response is False:
                            logging.info("Continuing despite mismatches as per user request.")
                            alert_disabled = True
                            state.low_confidence_prompt_disabled = True
                    if large_mismatch_hits >= restart_threshold:
                        logging.warning(
                            "Large board mismatch persisted for %d checks. Restarting session.",
                            large_mismatch_hits,
                        )
                        restart_points = list(current_points)
                        restart_coord_id = active_coord_id
                        break
                else:
                    large_mismatch_hits = 0
                    large_mismatch_prompt_shown = False
                    if not state.low_confidence_prompt_disabled:
                        alert_disabled = False

        except KeyboardInterrupt:
            logging.info("Interrupted by user. Cleaning up.")
            break
        finally:
            if "engine" in locals():
                try:
                    engine.close()
                except Exception:
                    logging.debug("Error while closing engine.", exc_info=True)
            cv2.destroyAllWindows()

        if restart_points is not None:
            cached_points = restart_points
            if restart_coord_id is not None:
                active_coord_id = restart_coord_id
                cache.mark_last_used(active_coord_id)
            continue
        break


class BotRunnerThread(threading.Thread):
    def __init__(self, config: CLIConfig) -> None:
        super().__init__(daemon=True)
        self.config = config
        self.stop_event = threading.Event()
        self.error: Optional[Exception] = None

    def run(self) -> None:
        try:
            run_bot(self.config, stop_event=self.stop_event)
        except Exception as exc:
            self.error = exc
            logging.exception("Screen Chess Bot encountered an error.", exc_info=True)

    def stop(self) -> None:
        self.stop_event.set()


class TkTextHandler(logging.Handler):
    def __init__(self, log_queue: "queue.Queue[str]") -> None:
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            self.handleError(record)
            return
        self.log_queue.put(msg)


class QueueStream(io.TextIOBase):
    def __init__(self, log_queue: "queue.Queue[str]", prefix: str = "") -> None:
        super().__init__()
        self.log_queue = log_queue
        self.prefix = prefix
        self._buffer = ""

    def write(self, message: str) -> int:
        if not isinstance(message, str):
            message = str(message)
        self._buffer += message
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line:
                self.log_queue.put(f"{self.prefix}{line}")
        return len(message)

    def flush(self) -> None:
        if self._buffer:
            self.log_queue.put(f"{self.prefix}{self._buffer}")
            self._buffer = ""


class BotUIApp:
    PREVIEW_SIZE = 220

    def __init__(self, base_config: Optional[CLIConfig] = None) -> None:
        self.base_config = base_config
        self.root = tk.Tk()
        self.root.wm_attributes("-topmost", True)
        self.root.title("Screen Chess Bot UI")
        self.cache = BoardCoordinateCache()
        self.bot_thread: Optional[BotRunnerThread] = None
        self.selection_thread: Optional[threading.Thread] = None
        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.log_handler = TkTextHandler(self.log_queue)
        self.log_handler.setFormatter(
            logging.Formatter("%H:%M:%S [%(levelname)s] %(message)s")
        )
        logging.getLogger().addHandler(self.log_handler)
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self.stdout_redirect = QueueStream(self.log_queue)
        self.stderr_redirect = QueueStream(self.log_queue, prefix="[stderr] ")
        sys.stdout = self.stdout_redirect
        sys.stderr = self.stderr_redirect
        self.preview_photo: Optional[ImageTk.PhotoImage] = None
        self.coord_ids: List[str] = []
        self.selected_coord_id: Optional[str] = None

        self._build_ui()
        self._refresh_coord_list()
        if self.selected_coord_id and self.selected_coord_id in self.coord_ids:
            index = self.coord_ids.index(self.selected_coord_id)
            self.coord_listbox.selection_set(index)
        self._update_preview(auto=True)
        self.status_var.set("Idle")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(1000, self._refresh_preview_loop)
        self.root.after(200, self._poll_log_queue)
        logging.info("UI ready. Configure options and press Start to run the bot.")

    def run(self) -> None:
        self.root.mainloop()

    def _build_ui(self) -> None:
        base = self.base_config
        engine_path = base.engine_path if base and base.engine_path else ""
        yolo_path = base.yolo_weights if base and base.yolo_weights else ""
        stockfish_range = base.stockfish_power_range if base else None
        move_delay_range = base.move_delay_range_s if base else None
        selected_coord = str(base.use_cached_coord) if base and base.use_cached_coord is not None else None

        self.selected_coord_id = selected_coord

        self.engine_path_var = tk.StringVar(value=engine_path)
        self.yolo_path_var = tk.StringVar(value=yolo_path)
        self.power_min_var = tk.StringVar(value=str(stockfish_range[0]) if stockfish_range else "")
        self.power_max_var = tk.StringVar(value=str(stockfish_range[1]) if stockfish_range else "")
        self.delay_min_var = tk.StringVar(value=str(move_delay_range[0]) if move_delay_range else "")
        self.delay_max_var = tk.StringVar(value=str(move_delay_range[1]) if move_delay_range else "")
        self.show_debug_var = tk.BooleanVar(value=base.show_debug if base else False)
        self.use_cached_var = tk.BooleanVar(value=selected_coord is not None)
        self.show_move_only_var = tk.BooleanVar(value=base.show_move_only if base else False)
        self.hotkey_mode_var = tk.BooleanVar(value=base.single_move_hotkeys if base else False)

        main = tk.Frame(self.root, padx=10, pady=10)
        main.pack(fill="both", expand=True)
        main.grid_columnconfigure(0, weight=1)
        main.grid_columnconfigure(1, weight=1)
        main.grid_rowconfigure(2, weight=1)

        config_frame = tk.LabelFrame(main, text="Configuration", padx=8, pady=8)
        config_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=(0, 8))
        config_frame.grid_columnconfigure(1, weight=1)
        config_frame.grid_columnconfigure(2, weight=1)

        tk.Label(config_frame, text="Stockfish path:").grid(row=0, column=0, sticky="e", pady=2)
        tk.Entry(config_frame, textvariable=self.engine_path_var).grid(row=0, column=1, columnspan=2, sticky="ew", padx=4, pady=2)

        tk.Label(config_frame, text="YOLO weights:").grid(row=1, column=0, sticky="e", pady=2)
        tk.Entry(config_frame, textvariable=self.yolo_path_var).grid(row=1, column=1, columnspan=2, sticky="ew", padx=4, pady=2)

        tk.Label(config_frame, text="Stockfish power (min/max):").grid(row=2, column=0, sticky="e", pady=2)
        tk.Entry(config_frame, width=6, textvariable=self.power_min_var).grid(row=2, column=1, sticky="w", padx=4, pady=2)
        tk.Entry(config_frame, width=6, textvariable=self.power_max_var).grid(row=2, column=2, sticky="w", padx=4, pady=2)

        tk.Label(config_frame, text="Move delay (min/max):").grid(row=3, column=0, sticky="e", pady=2)
        tk.Entry(config_frame, width=6, textvariable=self.delay_min_var).grid(row=3, column=1, sticky="w", padx=4, pady=2)
        tk.Entry(config_frame, width=6, textvariable=self.delay_max_var).grid(row=3, column=2, sticky="w", padx=4, pady=2)

        tk.Checkbutton(
            config_frame,
            text="Show debug windows",
            variable=self.show_debug_var,
            state="disabled",
        ).grid(row=4, column=0, columnspan=3, sticky="w", padx=4, pady=(6, 2))
        tk.Checkbutton(
            config_frame,
            text="Suggest moves only",
            variable=self.show_move_only_var,
        ).grid(row=5, column=0, columnspan=3, sticky="w", padx=4, pady=(2, 2))
        tk.Checkbutton(
            config_frame,
            text="Single-move hotkeys (D suggest, S auto)",
            variable=self.hotkey_mode_var,
        ).grid(row=6, column=0, columnspan=3, sticky="w", padx=4, pady=(2, 2))

        coord_frame = tk.LabelFrame(main, text="Cached Coordinates", padx=8, pady=8)
        coord_frame.grid(row=0, column=1, sticky="nsew", pady=(0, 8))
        coord_frame.grid_columnconfigure(0, weight=1)

        self.use_cached_check = tk.Checkbutton(coord_frame, text="Use cached coord", variable=self.use_cached_var)
        self.use_cached_check.grid(row=0, column=0, sticky="w")

        self.coord_listbox = tk.Listbox(coord_frame, height=6, exportselection=False)
        self.coord_listbox.grid(row=1, column=0, sticky="nsew", pady=4)
        self.coord_listbox.bind("<<ListboxSelect>>", self._on_coord_select)

        coord_buttons = tk.Frame(coord_frame)
        coord_buttons.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        coord_buttons.grid_columnconfigure(0, weight=1)
        coord_buttons.grid_columnconfigure(1, weight=1)
        coord_buttons.grid_columnconfigure(2, weight=1)

        tk.Button(coord_buttons, text="Use cached coord", command=self._apply_selected_coord).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        tk.Button(coord_buttons, text="Remove selected", command=self._remove_coord).grid(row=0, column=1, sticky="ew", padx=(4, 4))
        self.select_button = tk.Button(coord_buttons, text="Select board…", command=self._select_new_board)
        self.select_button.grid(row=0, column=2, sticky="ew")

        preview_frame = tk.LabelFrame(main, text="Board Preview", padx=8, pady=8)
        preview_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")
        preview_frame.grid_columnconfigure(0, weight=1)
        preview_frame.grid_rowconfigure(0, weight=1)
        self.preview_canvas = tk.Canvas(
            preview_frame,
            width=self.PREVIEW_SIZE,
            height=self.PREVIEW_SIZE,
            bg="#181818",
            highlightthickness=1,
            relief="sunken",
        )
        self.preview_canvas.grid(row=0, column=0, sticky="n")
        self.preview_canvas_text = self.preview_canvas.create_text(
            self.PREVIEW_SIZE // 2,
            self.PREVIEW_SIZE // 2,
            text="No preview available.",
            fill="#d0d0d0",
        )

        console_frame = tk.LabelFrame(main, text="Console", padx=6, pady=6)
        console_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(8, 8))
        console_frame.grid_columnconfigure(0, weight=1)
        console_frame.grid_rowconfigure(0, weight=1)
        self.log_text = tk.Text(
            console_frame,
            height=10,
            state="disabled",
            wrap="word",
            bg="#101010",
            fg="#f0f0f0",
            insertbackground="#f0f0f0",
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_scroll = tk.Scrollbar(console_frame, orient="vertical", command=self.log_text.yview)
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=log_scroll.set)

        control_frame = tk.Frame(main, pady=8)
        control_frame.grid(row=3, column=0, columnspan=2, sticky="ew")
        control_frame.grid_columnconfigure(0, weight=1)
        control_frame.grid_columnconfigure(1, weight=1)
        control_frame.grid_columnconfigure(2, weight=1)
        control_frame.grid_columnconfigure(3, weight=1)

        self.start_button = tk.Button(control_frame, text="Start", command=self._start_bot)
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.stop_button = tk.Button(control_frame, text="Stop", command=self._stop_bot, state="disabled")
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=(4, 4))
        tk.Button(control_frame, text="Apply settings", command=self._apply_settings).grid(row=0, column=2, sticky="ew", padx=(4, 4))
        tk.Button(control_frame, text="Refresh preview", command=self._update_preview).grid(row=0, column=3, sticky="ew")

        self.status_var = tk.StringVar(value="Idle")
        tk.Label(main, textvariable=self.status_var, anchor="w").grid(row=4, column=0, columnspan=2, sticky="ew")

    def _refresh_preview_loop(self) -> None:
        if self.selected_coord_id is not None:
            self._update_preview(auto=True)
        self.root.after(1500, self._refresh_preview_loop)

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, message + "\n")
        line_count = int(self.log_text.index("end-1c").split(".")[0])
        if line_count > 1000:
            excess = line_count - 1000
            self.log_text.delete("1.0", f"{excess + 1}.0")
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")

    def _poll_log_queue(self) -> None:
        try:
            while True:
                entry = self.log_queue.get_nowait()
                self._append_log(entry)
        except queue.Empty:
            pass
        finally:
            self.root.after(200, self._poll_log_queue)

    def _current_selection_index(self) -> Optional[int]:
        selection = self.coord_listbox.curselection()
        if not selection:
            return None
        return selection[0]

    def _on_coord_select(self, _event: Optional[object] = None) -> None:
        index = self._current_selection_index()
        if index is None or index >= len(self.coord_ids):
            self.selected_coord_id = None
            self._update_preview()
            return
        self.selected_coord_id = self.coord_ids[index]
        self._update_preview()

    def _apply_selected_coord(self) -> None:
        if self.selected_coord_id is None:
            messagebox.showinfo("Chess Bot", "Select a cached coordinate first.")
            return
        self.use_cached_var.set(True)
        self.status_var.set(f"Using cached coord {self.selected_coord_id}")

    def _remove_coord(self) -> None:
        index = self._current_selection_index()
        if index is None or index >= len(self.coord_ids):
            messagebox.showinfo("Chess Bot", "Select a cached coordinate to remove.")
            return
        coord_id = self.coord_ids[index]
        if not messagebox.askyesno("Confirm removal", f"Remove cached coordinate {coord_id}?"):
            return
        if self.cache.delete(coord_id):
            if self.selected_coord_id == coord_id:
                self.selected_coord_id = None
            self._refresh_coord_list()
            self._update_preview()
            logging.info("Removed cached coordinate id %s.", coord_id)
        else:
            messagebox.showwarning("Chess Bot", f"Unable to remove coordinate {coord_id}.")

    def _select_new_board(self) -> None:
        if self.selection_thread and self.selection_thread.is_alive():
            messagebox.showinfo("Chess Bot", "Board selection already in progress.")
            return
        self.status_var.set("Launching board selector...")
        logging.info("Launching board selector.")
        self.select_button.config(state="disabled")
        self.selection_thread = threading.Thread(target=self._run_board_selection, daemon=True)
        self.selection_thread.start()

    def _run_board_selection(self) -> None:
        result: Optional[BoardSelectionResult] = None
        error: Optional[Exception] = None
        try:
            selector = ScreenSelector()
            full_screen, origin = take_full_screenshot()
            result = selector.select_board(full_screen, origin)
        except Exception as exc:
            error = exc
            logging.exception("Board selection failed.", exc_info=True)
        finally:
            cv2.destroyAllWindows()
            self.root.after(0, lambda: self._handle_selection_result(result, error))

    def _handle_selection_result(
        self,
        result: Optional[BoardSelectionResult],
        error: Optional[Exception],
    ) -> None:
        self.select_button.config(state="normal")
        self.selection_thread = None

        if error is not None:
            messagebox.showerror("Chess Bot", f"Board selection failed:\n{error}")
            self.status_var.set("Board selection failed.")
            logging.error("Board selection failed: %s", error)
            return

        if result is None:
            self.status_var.set("Board selection canceled.")
            logging.info("Board selection canceled by user.")
            return

        points = [(int(x), int(y)) for x, y in result.points]
        image = self._grab_region(points)
        if image is not None:
            self._render_preview_image(image)

        save_points = result.save_requested
        coord_id_hint: Optional[str] = self.selected_coord_id
        if not save_points:
            save_points = messagebox.askyesno(
                "Chess Bot",
                "Save the selected coordinates to cache?",
            )
        if not save_points:
            self.status_var.set("Coordinates captured (not saved).")
            logging.info("Captured coordinates without saving.")
            return

        try:
            saved_id = self.cache.save(points, coord_id=coord_id_hint, mark_last=True)
        except ValueError as exc:
            messagebox.showerror("Chess Bot", f"Unable to save coordinates:\n{exc}")
            self.status_var.set("Save failed.")
            logging.error("Failed to save coordinates: %s", exc)
            return

        self.selected_coord_id = saved_id
        self.use_cached_var.set(True)
        self._refresh_coord_list()
        self._update_preview()
        self.status_var.set(f"Saved coordinates as id {saved_id}.")
        logging.info("Saved board coordinates as id %s.", saved_id)

    def _refresh_coord_list(self) -> None:
        self.coord_ids = self.cache.available_ids()
        self.coord_listbox.delete(0, tk.END)
        for coord_id in self.coord_ids:
            label = coord_id
            if self.cache.last_used == coord_id:
                label += " (last used)"
            self.coord_listbox.insert(tk.END, label)
        if self.selected_coord_id and self.selected_coord_id in self.coord_ids:
            idx = self.coord_ids.index(self.selected_coord_id)
            self.coord_listbox.selection_set(idx)
        else:
            self.coord_listbox.selection_clear(0, tk.END)

    def _update_preview(self, auto: bool = False) -> None:
        coord_id = self.selected_coord_id
        if coord_id is None or coord_id not in self.cache.coords:
            if not auto:
                self._clear_preview("No cached coord selected.")
            else:
                self._clear_preview()
            return
        image = self._capture_preview_image(coord_id)
        if image is None:
            if not auto:
                self._clear_preview("Preview unavailable.")
            return
        self._render_preview_image(image)

    def _clear_preview(self, message: str = "No preview available.") -> None:
        self.preview_photo = None
        self.preview_canvas.delete("all")
        self.preview_canvas.create_text(
            self.PREVIEW_SIZE // 2,
            self.PREVIEW_SIZE // 2,
            text=message,
            fill="#d0d0d0",
        )

    def _render_preview_image(self, image: np.ndarray) -> None:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        width, height = pil_img.size
        if width <= 0 or height <= 0:
            self._clear_preview("Invalid preview data.")
            return
        aspect = width / height
        if aspect >= 1.0:
            new_w = self.PREVIEW_SIZE
            new_h = max(1, int(self.PREVIEW_SIZE / max(aspect, 1e-6)))
        else:
            new_h = self.PREVIEW_SIZE
            new_w = max(1, int(self.PREVIEW_SIZE * max(aspect, 1e-6)))
        resized = pil_img.resize((new_w, new_h), Image.LANCZOS)
        canvas = Image.new("RGB", (self.PREVIEW_SIZE, self.PREVIEW_SIZE), color=(24, 24, 24))
        offset = ((self.PREVIEW_SIZE - new_w) // 2, (self.PREVIEW_SIZE - new_h) // 2)
        canvas.paste(resized, offset)
        self.preview_photo = ImageTk.PhotoImage(canvas)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(self.PREVIEW_SIZE // 2, self.PREVIEW_SIZE // 2, image=self.preview_photo)

    def _capture_preview_image(self, coord_id: str) -> Optional[np.ndarray]:
        raw_points = self.cache.coords.get(coord_id)
        if not raw_points:
            return None
        points = [(int(p[0]), int(p[1])) for p in raw_points]
        return self._grab_region(points)

    def _grab_region(self, points: Sequence[Tuple[int, int]]) -> Optional[np.ndarray]:
        try:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            left = min(xs)
            top = min(ys)
            right = max(xs)
            bottom = max(ys)
            width = max(right - left, 1)
            height = max(bottom - top, 1)
            bbox = {"left": left, "top": top, "width": width, "height": height}
            with mss.mss() as capture:
                raw = capture.grab(bbox)
        except Exception:
            logging.debug("Failed to capture preview region.", exc_info=True)
            return None
        return ensure_channel_order(np.array(raw))

    def _start_bot(self) -> None:
        if self.bot_thread and self.bot_thread.is_alive():
            messagebox.showinfo("Chess Bot", "Bot is already running.")
            return
        try:
            config = self._build_config()
        except ValueError as exc:
            messagebox.showerror("Chess Bot", str(exc))
            return
        self.bot_thread = BotRunnerThread(config)
        self.bot_thread.start()
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.status_var.set("Running")
        logging.info("Bot started via UI.")
        self.root.after(500, self._poll_bot_thread)

    def _stop_bot(self) -> None:
        if not self.bot_thread:
            return
        self.bot_thread.stop()
        self.stop_button.config(state="disabled")
        self.status_var.set("Stopping...")
        logging.info("Stop requested from UI.")

    def _poll_bot_thread(self) -> None:
        if not self.bot_thread:
            return
        if self.bot_thread.is_alive():
            self.root.after(500, self._poll_bot_thread)
            return
        error = self.bot_thread.error
        self.bot_thread = None
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.status_var.set("Idle")
        if error:
            messagebox.showerror("Chess Bot", f"Bot stopped due to an error:\n{error}")

    def _build_config(self) -> CLIConfig:
        engine_path = self.engine_path_var.get().strip()
        yolo_path = self.yolo_path_var.get().strip()
        if not yolo_path:
            raise ValueError("YOLO weights path is required.")
        move_delay = self._parse_float_range(self.delay_min_var.get(), self.delay_max_var.get())
        stockfish_range = self._parse_int_range(self.power_min_var.get(), self.power_max_var.get())
        coord_id: Optional[int] = None
        if self.use_cached_var.get():
            if self.selected_coord_id is None:
                raise ValueError("Select a cached coordinate or disable 'Use cached coord'.")
            coord_id = int(self.selected_coord_id)

        return CLIConfig(
            engine_path=engine_path or None,
            move_time_ms=None,
            move_delay_range_s=move_delay,
            stockfish_power_range=stockfish_range,
            use_cached_coord=coord_id,
            launch_ui=False,
            depth=12,
            color="auto",
            show_debug=self.show_debug_var.get(),
            show_move_only=self.show_move_only_var.get(),
            single_move_hotkeys=self.hotkey_mode_var.get(),
            promotion="q",
            yolo_weights=yolo_path,
            yolo_confidence=0.35,
            yolo_iou=0.45,
            yolo_device=None,
            trust_detections=False,
        )

    def _parse_float_range(self, low_str: str, high_str: str) -> Optional[Tuple[float, float]]:
        low_str = low_str.strip()
        high_str = high_str.strip()
        if not low_str and not high_str:
            return None
        if not low_str or not high_str:
            raise ValueError("Provide both min and max values for move delay or leave both empty.")
        try:
            low = float(low_str)
            high = float(high_str)
        except ValueError as exc:
            raise ValueError("Move delay values must be numeric.") from exc
        if high < low:
            raise ValueError("Move delay max must be greater than or equal to min.")
        return (low, high)

    def _parse_int_range(self, low_str: str, high_str: str) -> Optional[Tuple[int, int]]:
        low_str = low_str.strip()
        high_str = high_str.strip()
        if not low_str and not high_str:
            return None
        if not low_str or not high_str:
            raise ValueError("Provide both min and max values for Stockfish power or leave both empty.")
        try:
            low = int(low_str)
            high = int(high_str)
        except ValueError as exc:
            raise ValueError("Stockfish power values must be integers.") from exc
        if not (0 <= low <= 20 and 0 <= high <= 20):
            raise ValueError("Stockfish power values must be between 0 and 20.")
        if high < low:
            raise ValueError("Stockfish power max must be greater than or equal to min.")
        return (low, high)

    def _apply_settings(self) -> None:
        try:
            move_delay = self._parse_float_range(self.delay_min_var.get(), self.delay_max_var.get())
            stockfish_range = self._parse_int_range(self.power_min_var.get(), self.power_max_var.get())
        except ValueError as exc:
            messagebox.showerror("Chess Bot", str(exc))
            return

        show_move_only = self.show_move_only_var.get()
        single_move_hotkeys = self.hotkey_mode_var.get()

        if self.base_config is not None:
            self.base_config.move_delay_range_s = move_delay
            self.base_config.stockfish_power_range = stockfish_range
            self.base_config.show_move_only = show_move_only
            self.base_config.single_move_hotkeys = single_move_hotkeys

        if self.bot_thread and self.bot_thread.is_alive():
            cfg = self.bot_thread.config
            cfg.move_delay_range_s = move_delay
            cfg.stockfish_power_range = stockfish_range
            cfg.show_move_only = show_move_only
            cfg.single_move_hotkeys = single_move_hotkeys
            logging.info("Applied move delay and Stockfish power settings to running bot.")
            self.status_var.set("Settings applied to running bot.")
        else:
            logging.info("Updated settings; they will apply on the next start.")
            self.status_var.set("Settings updated for next start.")

    def _remove_log_handler(self) -> None:
        if getattr(self, "log_handler", None):
            logging.getLogger().removeHandler(self.log_handler)
            self.log_handler = None
        if getattr(self, "_original_stdout", None) is not None:
            self.stdout_redirect.flush()
            sys.stdout = self._original_stdout
            self._original_stdout = None
        if getattr(self, "_original_stderr", None) is not None:
            self.stderr_redirect.flush()
            sys.stderr = self._original_stderr
            self._original_stderr = None

    def _on_close(self) -> None:
        if self.bot_thread and self.bot_thread.is_alive():
            self._stop_bot()
            self.root.after(200, self._await_shutdown)
        else:
            self._remove_log_handler()
            self.root.destroy()

    def _await_shutdown(self) -> None:
        if self.bot_thread and self.bot_thread.is_alive():
            self.root.after(200, self._await_shutdown)
            return
        self._remove_log_handler()
        self.root.destroy()


def launch_ui(base_config: Optional[CLIConfig] = None) -> None:
    app = BotUIApp(base_config)
    app.run()


def parse_args(argv: Optional[Sequence[str]] = None) -> CLIConfig:
    parser = argparse.ArgumentParser(description="Screen-based chess bot powered by Stockfish.")
    parser.add_argument("--engine-path", type=str, default=None, help="Path to Stockfish binary.")
    parser.add_argument("--move-time-ms", type=int, default=None, help="Search time per move (ms).")
    parser.add_argument(
        "--move-delay-range",
        type=float,
        nargs=2,
        metavar=("MIN_SEC", "MAX_SEC"),
        default=None,
        help="Randomized delay range in seconds before executing a move (min max).",
    )
    parser.add_argument("--depth", type=int, default=12, help="Search depth when time is not set.")
    parser.add_argument(
        "--stockfish-power",
        type=int,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="Randomize Stockfish skill level between MIN and MAX (inclusive). Valid range 0-20.",
    )
    parser.add_argument(
        "--use-cached-coord",
        type=int,
        default=None,
        help="Reuse a saved board coordinate set by id (saved via 's' in selector).",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch the graphical interface instead of running the bot from the CLI.",
    )
    parser.add_argument(
        "--show-move-only",
        action="store_true",
        help="Preview moves on screen instead of executing them automatically.",
    )
    parser.add_argument(
        "--single-move-hotkeys",
        action="store_true",
        help="Enable manual hotkeys (press 'd' to preview a single move, 's' to play one move automatically).",
    )
    parser.add_argument(
        "--color",
        type=str,
        default="auto",
        choices=["white", "black", "auto"],
        help="Which side we play. 'auto' infers from initial position.",
    )
    parser.add_argument(
        "--show-debug",
        type=int,
        default=0,
        choices=[0, 1],
        help="Show debug windows and save warped boards.",
    )
    parser.add_argument(
        "--trust-detections",
        action="store_true",
        help="When set, prefer overriding the internal board with detection results if reconciliation fails.",
    )
    parser.add_argument(
        "--promotion",
        type=str,
        default="q",
        choices=["q", "r", "b", "n"],
        help="Preferred promotion piece.",
    )
    parser.add_argument(
        "--yolo-weights",
        type=str,
        default=None,
        help="Path to a custom-trained Ultralytics YOLO weights file (.pt).",
    )
    parser.add_argument(
        "--yolo-confidence",
        type=float,
        default=0.35,
        help="Minimum confidence threshold (0-1) for YOLO detections.",
    )
    parser.add_argument(
        "--yolo-iou",
        type=float,
        default=0.45,
        help="IoU threshold (0-1) used by YOLO non-max suppression.",
    )
    parser.add_argument(
        "--yolo-device",
        type=str,
        default=None,
        help="Execution device for YOLO (e.g. 'cuda:0', 'cpu'). Defaults to autoselect.",
    )
    args = parser.parse_args(argv)
    config = CLIConfig(
        engine_path=args.engine_path,
        move_time_ms=args.move_time_ms,
        move_delay_range_s=tuple(args.move_delay_range) if args.move_delay_range else None,
        stockfish_power_range=tuple(args.stockfish_power) if args.stockfish_power else None,
        use_cached_coord=args.use_cached_coord,
        launch_ui=bool(args.ui),
        show_move_only=bool(args.show_move_only),
        depth=args.depth,
        color=args.color,
        show_debug=bool(args.show_debug),
        trust_detections=bool(args.trust_detections),
        promotion=args.promotion,
        yolo_weights=args.yolo_weights,
        yolo_confidence=args.yolo_confidence,
        yolo_iou=args.yolo_iou,
        yolo_device=args.yolo_device,
        single_move_hotkeys=bool(args.single_move_hotkeys),
    )
    return config


def setup_logging(debug: bool, console: bool = True) -> None:
    level = logging.DEBUG if debug else logging.INFO
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    handlers: List[logging.Handler] = []
    if console:
        stream_handler = logging.StreamHandler(sys.stdout)
        handlers.append(stream_handler)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    config = parse_args(argv)
    setup_logging(config.show_debug, console=not config.launch_ui)
    if config.launch_ui:
        launch_ui(config)
        return
    run_bot(config)


if __name__ == "__main__":
    main()


class TemplateManager:
    def __init__(self, template_dir: Path, cell_size: int, margin: int) -> None:
        self.template_dir = template_dir
        self.cell_size = cell_size
        self.crop_size = cell_size - 2 * margin
        self.templates: Dict[str, Template] = {}
        self._load_templates()

    def _load_templates(self) -> None:
        if not self.template_dir.exists():
            logging.warning(
                "Template directory %s not found. Falling back to synthetic templates.",
                self.template_dir,
            )
            self._generate_synthetic_templates()
            return

        loaded_any = False
        for label, filename in TEMPLATE_FILENAMES.items():
            path = self.template_dir / filename
            if not path.exists():
                logging.warning("Missing template %s (%s). Will synthesise.", label, filename)
                continue
            template_bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if template_bgr is None:
                logging.warning("Failed to read template %s. Using synthetic fallback.", path)
                continue
            gray = self._prepare_template_image(template_bgr)
            edges = preprocess_edges(gray)
            resized = cv2.resize(edges, (self.crop_size, self.crop_size), interpolation=cv2.INTER_AREA)
            self.templates[label] = Template(label=label, original=gray, edges=resized)
            loaded_any = True

        if not loaded_any:
            logging.warning("No templates loaded from disk. Using synthetic defaults.")
            self._generate_synthetic_templates()
        else:
            missing = {label for label in TEMPLATE_FILENAMES if label not in self.templates}
            if missing:
                logging.info("Synthesising missing templates for %s", ", ".join(sorted(missing)))
                self._generate_synthetic_templates(target_labels=missing)

    def _generate_synthetic_templates(self, target_labels: Optional[Iterable[str]] = None) -> None:
        labels = target_labels or TEMPLATE_FILENAMES.keys()
        for label in labels:
            blank = np.zeros((self.cell_size, self.cell_size, 3), dtype=np.uint8)
            cv2.rectangle(blank, (0, 0), (self.cell_size - 1, self.cell_size - 1), TEMPLATE_COLOR, thickness=2)
            text = label[1].upper()
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 2.4
            thickness = 4
            text_size = cv2.getTextSize(text, font, scale, thickness)[0]
            text_x = (self.cell_size - text_size[0]) // 2
            text_y = (self.cell_size + text_size[1]) // 2
            cv2.putText(blank, text, (text_x, text_y), font, scale, TEMPLATE_COLOR, thickness, cv2.LINE_AA)
            gray = cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY)
            edges = preprocess_edges(gray)
            resized = cv2.resize(edges, (self.crop_size, self.crop_size), interpolation=cv2.INTER_AREA)
            self.templates[label] = Template(label=label, original=gray, edges=resized)

    @staticmethod
    def _prepare_template_image(img: np.ndarray) -> np.ndarray:
        if img.shape[2] == 4:
            alpha = img[:, :, 3]
            bgr = img[:, :, :3]
            bg = np.full_like(bgr, TEMPLATE_BG, dtype=np.uint8)
            alpha_norm = alpha[:, :, None].astype(np.float32) / 255.0
            composed = (bgr.astype(np.float32) * alpha_norm + bg.astype(np.float32) * (1 - alpha_norm)).astype(
                np.uint8
            )
            gray = cv2.cvtColor(composed, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray

    def best_match(self, square_edges: np.ndarray) -> SquareDetection:
        best_label = DEFAULT_PIECE_LABEL
        best_score = -1.0
        for label, template in self.templates.items():
            result = cv2.matchTemplate(square_edges, template.edges, cv2.TM_CCOEFF_NORMED)
            score = float(result[0, 0])
            if score > best_score:
                best_label = label
                best_score = score
        return SquareDetection(label=best_label, score=best_score)
