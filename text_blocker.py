#!/usr/bin/env python3
"""
Text Blocker Lite - OCR-based video text censoring with low resource usage.
Samples frames at a low rate, optionally skips OCR on similar frames, and
builds time-ranged drawbox filters for ffmpeg.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import easyocr
from PIL import Image
import imagehash


def _parse_fps(rate: str) -> float:
    if not rate:
        return 30.0
    if "/" in rate:
        num, denom = rate.split("/", 1)
        try:
            num_f = float(num)
            denom_f = float(denom)
            return num_f / denom_f if denom_f else 30.0
        except ValueError:
            return 30.0
    try:
        return float(rate)
    except ValueError:
        return 30.0


def get_video_info(video_path: str) -> dict:
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_format", video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)

    video_stream = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    if not video_stream:
        raise RuntimeError("No video stream found")

    fps = _parse_fps(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate"))
    bitrate = (
        video_stream.get("bit_rate")
        or data.get("format", {}).get("bit_rate")
        or "5000000"
    )

    return {
        "width": int(video_stream["width"]),
        "height": int(video_stream["height"]),
        "fps": fps or 30.0,
        "fps_rational": video_stream.get("r_frame_rate", "30/1"),
        "duration": float(data.get("format", {}).get("duration", 0)),
        "bitrate": int(float(bitrate)),
        "codec": video_stream.get("codec_name", "h264"),
        "pix_fmt": video_stream.get("pix_fmt", "yuv420p"),
    }


def compute_frame_hash(frame: np.ndarray) -> imagehash.ImageHash:
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return imagehash.phash(pil_image, hash_size=8)


def detect_boxes(
    reader: easyocr.Reader,
    img: np.ndarray,
    scale_x: float,
    scale_y: float,
    original_width: int,
    original_height: int,
    padding: int,
) -> list[tuple[int, int, int, int]]:
    horizontal, free_form = reader.detect(img)
    boxes = []

    if horizontal and horizontal[0] is not None:
        for box in horizontal[0]:
            x_min, x_max, y_min, y_max = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            x = int(x_min * scale_x) - padding
            y = int(y_min * scale_y) - padding
            w = int((x_max - x_min) * scale_x) + padding * 2
            h = int((y_max - y_min) * scale_y) + padding * 2
            x, y = max(0, x), max(0, y)
            w = min(w, original_width - x)
            h = min(h, original_height - y)
            if w > 0 and h > 0:
                boxes.append((x, y, w, h))

    if free_form and free_form[0] is not None:
        for poly in free_form[0]:
            pts = np.array(poly).reshape(-1, 2)
            x_min, y_min = pts.min(axis=0)
            x_max, y_max = pts.max(axis=0)
            x = int(x_min * scale_x) - padding
            y = int(y_min * scale_y) - padding
            w = int((x_max - x_min) * scale_x) + padding * 2
            h = int((y_max - y_min) * scale_y) + padding * 2
            x, y = max(0, x), max(0, y)
            w = min(w, original_width - x)
            h = min(h, original_height - y)
            if w > 0 and h > 0:
                boxes.append((x, y, w, h))

    return boxes


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (
        ax + aw + pad < bx
        or bx + bw + pad < ax
        or ay + ah + pad < by
        or by + bh + pad < ay
    )


def merge_boxes(boxes: list[tuple[int, int, int, int]], merge_pad: int) -> list[tuple[int, int, int, int]]:
    merged: list[tuple[int, int, int, int]] = []
    for box in sorted(boxes, key=lambda b: (b[1], b[0], b[2], b[3])):
        x, y, w, h = box
        changed = True
        while changed:
            changed = False
            new_merged = []
            for other in merged:
                if _boxes_overlap((x, y, w, h), other, merge_pad):
                    ox, oy, ow, oh = other
                    x1 = min(x, ox)
                    y1 = min(y, oy)
                    x2 = max(x + w, ox + ow)
                    y2 = max(y + h, oy + oh)
                    x, y = x1, y1
                    w, h = x2 - x1, y2 - y1
                    changed = True
                else:
                    new_merged.append(other)
            merged = new_merged
        merged.append((x, y, w, h))
    return merged


def compress_ranges(ranges: list[tuple[float, float, list[tuple[int, int, int, int]]]]) -> list:
    if not ranges:
        return []

    compressed = []
    current_start, current_end, current_boxes = ranges[0]
    current_key = tuple(sorted(current_boxes)) if current_boxes else None

    for start, end, boxes in ranges[1:]:
        key = tuple(sorted(boxes)) if boxes else None
        if key == current_key:
            current_end = end
        else:
            if current_boxes:
                compressed.append((current_start, current_end, current_boxes))
            current_start, current_end, current_boxes = start, end, boxes
            current_key = key

    if current_boxes:
        compressed.append((current_start, current_end, current_boxes))

    return compressed


def build_ffmpeg_filter(box_ranges: list[tuple[int, int, int, int, float, float]], max_filters: int) -> str:
    if not box_ranges:
        return "null"

    if len(box_ranges) > max_filters:
        # Fall back to union-of-all boxes for the full duration.
        max_end = max(end_t for _, _, _, _, _, end_t in box_ranges)
        union_boxes: dict[tuple[int, int, int, int], float] = {}
        for x, y, w, h, _, _ in box_ranges:
            union_boxes[(x, y, w, h)] = max_end
        box_ranges = [(x, y, w, h, 0.0, end_t) for (x, y, w, h), end_t in union_boxes.items()]
        if len(box_ranges) > max_filters:
            x1 = min(x for x, _, _, _, _, _ in box_ranges)
            y1 = min(y for _, y, _, _, _, _ in box_ranges)
            x2 = max(x + w for x, _, w, _, _, _ in box_ranges)
            y2 = max(y + h for _, y, _, h, _, _ in box_ranges)
            box_ranges = [(x1, y1, x2 - x1, y2 - y1, 0.0, max_end)]

    filters = []
    for x, y, w, h, start_t, end_t in box_ranges:
        if end_t <= start_t:
            continue
        enable = f"between(t,{start_t:.3f},{end_t:.3f})"
        filters.append(
            f"drawbox=x={x}:y={y}:w={w}:h={h}:c=black:t=fill:enable='{enable}'"
        )

    return ",".join(filters) if filters else "null"


def is_youtube_url(value: str) -> bool:
    lower = value.lower()
    return lower.startswith(("http://", "https://")) and ("youtube.com" in lower or "youtu.be" in lower)


def is_valid_youtube_id(value: str) -> bool:
    return re.match(r"^[A-Za-z0-9_-]{11}$", value) is not None


def sanitize_filename(value: str, max_len: int = 120) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = ascii_text.strip()
    ascii_text = re.sub(r"[^\w\-. ]+", "", ascii_text)
    ascii_text = re.sub(r"\s+", "_", ascii_text)
    ascii_text = re.sub(r"_+", "_", ascii_text).strip("._- ")
    if not ascii_text:
        return "video"
    return ascii_text[:max_len]


def build_output_filename(title: str, video_id: str) -> str:
    safe_title = sanitize_filename(title)
    if not safe_title or safe_title.lower() == video_id.lower():
        return f"{video_id}_blocked.mp4"
    return f"{safe_title}_{video_id}_blocked.mp4"


def resolve_youtube_metadata(url: str, verbose: bool) -> tuple[str, str]:
    cmd = ["yt-dlp", "--no-playlist", "--print", "%(id)s\t%(title)s", url]
    if not verbose:
        cmd += ["--quiet", "--no-warnings"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        err = result.stderr.strip() or "yt-dlp failed to resolve video id"
        raise RuntimeError(err)
    ids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not ids:
        raise RuntimeError("yt-dlp did not return a video id")
    last = ids[-1]
    if "\t" in last:
        video_id, title = last.split("\t", 1)
    else:
        video_id, title = last, ""
    video_id = video_id.strip()
    if not is_valid_youtube_id(video_id):
        raise RuntimeError(f"Invalid YouTube id: {video_id}")
    return video_id, title.strip()


def find_downloaded_file(temp_dir: str, video_id: str) -> Optional[Path]:
    candidates = [
        p for p in Path(temp_dir).glob(f"{video_id}.*")
        if p.is_file() and not p.name.endswith(".part")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_size)


def fetch_playlist_entries(playlist_url: str, verbose: bool) -> list[tuple[str, str]]:
    cmd = ["yt-dlp", "--flat-playlist", "--print", "%(id)s\t%(title)s", playlist_url]
    if not verbose:
        cmd += ["--quiet", "--no-warnings"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        err = result.stderr.strip() or "yt-dlp failed to fetch playlist"
        raise RuntimeError(err)
    entries = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        if "\t" in line:
            video_id, title = line.split("\t", 1)
        else:
            video_id, title = line, ""
        video_id = video_id.strip()
        title = title.strip()
        if not is_valid_youtube_id(video_id):
            if verbose:
                print(f"Skipping invalid video id: {video_id}")
            continue
        entries.append((video_id, title))
    return entries


def download_video(
    video_id: str,
    temp_dir: str,
    download_format: str,
    merge_format: Optional[str],
    verbose: bool,
) -> Optional[Path]:
    url = f"https://www.youtube.com/watch?v={video_id}"
    output_template = str(Path(temp_dir) / "%(id)s.%(ext)s")
    cmd = ["yt-dlp", "-f", download_format, "-o", output_template, url]
    if merge_format:
        cmd += ["--merge-output-format", merge_format]
    if not verbose:
        cmd += ["--quiet", "--no-warnings"]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        return None
    return find_downloaded_file(temp_dir, video_id)


def download_youtube_url(
    url: str,
    temp_dir: str,
    download_format: str,
    merge_format: Optional[str],
    verbose: bool,
) -> tuple[str, str, Optional[Path]]:
    video_id, title = resolve_youtube_metadata(url, verbose=verbose)
    return video_id, title, download_video(
        video_id,
        temp_dir=temp_dir,
        download_format=download_format,
        merge_format=merge_format,
        verbose=verbose,
    )


def process_playlist(
    playlist_url: str,
    output_dir: str,
    temp_dir: str,
    download_format: str,
    merge_format: Optional[str],
    keep_downloads: bool,
    skip_existing: bool,
    languages: list[str],
    ocr_height: int,
    sample_fps: float,
    padding: int,
    merge_pad: int,
    scene_threshold: int,
    skip_similar: bool,
    force_interval: float,
    max_filters: int,
    quality: str,
    verbose: bool,
) -> None:
    if shutil.which("yt-dlp") is None:
        raise RuntimeError("yt-dlp not found in PATH. Install with: pip install yt-dlp")

    output_path = Path(output_dir)
    temp_path = Path(temp_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    temp_path.mkdir(parents=True, exist_ok=True)

    entries = fetch_playlist_entries(playlist_url, verbose=verbose)
    if not entries:
        print("No videos found in playlist")
        return

    total = len(entries)
    for idx, (video_id, title) in enumerate(entries, 1):
        label = title if title else video_id
        print(f"\n=== Processing {idx}/{total}: {label} ({video_id}) ===")

        output_file = output_path / build_output_filename(title, video_id)
        if skip_existing and output_file.exists():
            print("Already processed, skipping...")
            continue

        print("Downloading...")
        input_file = download_video(
            video_id,
            temp_dir=temp_dir,
            download_format=download_format,
            merge_format=merge_format,
            verbose=verbose,
        )
        if input_file is None:
            print("Download failed, skipping...")
            continue

        try:
            print("Blocking text...")
            process_video(
                str(input_file),
                str(output_file),
                languages=languages,
                ocr_height=ocr_height,
                sample_fps=sample_fps,
                padding=padding,
                merge_pad=merge_pad,
                scene_threshold=scene_threshold,
                skip_similar=skip_similar,
                force_interval=force_interval,
                max_filters=max_filters,
                quality=quality,
                verbose=verbose,
            )
            print(f"Done: {output_file}")
        except Exception as exc:
            print(f"Error processing {video_id}: {exc}")
            if output_file.exists():
                output_file.unlink()
        finally:
            if not keep_downloads and input_file.exists():
                input_file.unlink()


def list_video_files(input_dir: str, recursive: bool) -> list[Path]:
    exts = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".flv", ".mpg", ".mpeg", ".m4v", ".3gp"}
    base = Path(input_dir)
    if recursive:
        items = base.rglob("*")
    else:
        items = base.glob("*")
    return sorted([p for p in items if p.is_file() and p.suffix.lower() in exts])


def process_folder(
    input_dir: str,
    output_dir: str,
    recursive: bool,
    skip_existing: bool,
    languages: list[str],
    ocr_height: int,
    sample_fps: float,
    padding: int,
    merge_pad: int,
    scene_threshold: int,
    skip_similar: bool,
    force_interval: float,
    max_filters: int,
    quality: str,
    verbose: bool,
) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    files = list_video_files(input_dir, recursive=recursive)
    if not files:
        print("No video files found")
        return

    total = len(files)
    for idx, video_path in enumerate(files, 1):
        relative = video_path.relative_to(input_path)
        output_parent = output_path / relative.parent
        output_parent.mkdir(parents=True, exist_ok=True)
        output_file = output_parent / f"{relative.stem}_blocked.mp4"

        print(f"\n=== Processing {idx}/{total}: {relative} ===")
        if skip_existing and output_file.exists():
            print("Already processed, skipping...")
            continue

        try:
            process_video(
                str(video_path),
                str(output_file),
                languages=languages,
                ocr_height=ocr_height,
                sample_fps=sample_fps,
                padding=padding,
                merge_pad=merge_pad,
                scene_threshold=scene_threshold,
                skip_similar=skip_similar,
                force_interval=force_interval,
                max_filters=max_filters,
                quality=quality,
                verbose=verbose,
            )
            print(f"Done: {output_file}")
        except Exception as exc:
            print(f"Error processing {video_path}: {exc}")
            if output_file.exists():
                output_file.unlink()


def process_youtube_video(
    url: str,
    output: str,
    temp_dir: str,
    download_format: str,
    merge_format: Optional[str],
    keep_downloads: bool,
    skip_existing: bool,
    languages: list[str],
    ocr_height: int,
    sample_fps: float,
    padding: int,
    merge_pad: int,
    scene_threshold: int,
    skip_similar: bool,
    force_interval: float,
    max_filters: int,
    quality: str,
    verbose: bool,
) -> None:
    if shutil.which("yt-dlp") is None:
        raise RuntimeError("yt-dlp not found in PATH. Install with: pip install yt-dlp")

    temp_path = Path(temp_dir)
    temp_path.mkdir(parents=True, exist_ok=True)

    output_path = Path(output)
    output_file: Path

    print("Downloading...")
    video_id, title, input_file = download_youtube_url(
        url,
        temp_dir=temp_dir,
        download_format=download_format,
        merge_format=merge_format,
        verbose=verbose,
    )
    if input_file is None:
        raise RuntimeError("Download failed")

    if output_path.exists() and output_path.is_dir():
        output_file = output_path / build_output_filename(title, video_id)
    else:
        output_file = output_path

    if skip_existing and output_file.exists():
        print("Already processed, skipping...")
        if not keep_downloads and input_file.exists():
            input_file.unlink()
        return

    try:
        print("Blocking text...")
        process_video(
            str(input_file),
            str(output_file),
            languages=languages,
            ocr_height=ocr_height,
            sample_fps=sample_fps,
            padding=padding,
            merge_pad=merge_pad,
            scene_threshold=scene_threshold,
            skip_similar=skip_similar,
            force_interval=force_interval,
            max_filters=max_filters,
            quality=quality,
            verbose=verbose,
        )
        print(f"Done: {output_file}")
    finally:
        if not keep_downloads and input_file.exists():
            input_file.unlink()


def process_video(
    input_path: str,
    output_path: str,
    languages: list[str],
    ocr_height: int,
    sample_fps: float,
    padding: int,
    merge_pad: int,
    scene_threshold: int,
    skip_similar: bool,
    force_interval: float,
    max_filters: int,
    quality: str,
    verbose: bool,
):
    info = get_video_info(input_path)
    original_width = info["width"]
    original_height = info["height"]
    fps = info["fps"] or 30.0

    if ocr_height > original_height:
        ocr_height = original_height

    if verbose:
        print(f"Video: {original_width}x{original_height} @ {fps:.2f} fps")
        print(f"Duration: {info['duration']:.1f}s")

    ocr_width = max(1, int(original_width * (ocr_height / original_height)))
    scale_x = original_width / ocr_width
    scale_y = original_height / ocr_height

    frame_interval = max(1, int(round(fps / sample_fps)))
    actual_sample_fps = fps / frame_interval

    if verbose:
        print(f"OCR resolution: {ocr_width}x{ocr_height}")
        print(f"Sampling: every {frame_interval} frames (~{actual_sample_fps:.2f} fps)")

    with tempfile.TemporaryDirectory() as temp_dir:
        extract_cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-vf", f"select='not(mod(n\\,{frame_interval}))',scale={ocr_width}:{ocr_height}",
            "-vsync", "vfr",
            "-q:v", "2",
            f"{temp_dir}/frame_%06d.jpg",
        ]
        if not verbose:
            extract_cmd.insert(1, "-loglevel")
            extract_cmd.insert(2, "error")
        subprocess.run(extract_cmd, check=True)

        frame_files = sorted(Path(temp_dir).glob("frame_*.jpg"))
        if verbose:
            print(f"Extracted {len(frame_files)} frames for OCR")

        if not frame_files:
            subprocess.run(["cp", input_path, output_path], check=True)
            return

        reader = easyocr.Reader(languages, gpu=False, verbose=False)

        sample_boxes: list[list[tuple[int, int, int, int]]] = []
        last_hash = None
        last_boxes: list[tuple[int, int, int, int]] = []
        last_ocr_index = -1
        force_interval_frames = int(round(force_interval * actual_sample_fps)) if force_interval > 0 else 0

        for idx, frame_path in enumerate(frame_files):
            img = cv2.imread(str(frame_path))
            if img is None:
                sample_boxes.append([])
                continue

            frame_hash = compute_frame_hash(img)
            should_ocr = True

            if skip_similar and last_hash is not None:
                hash_diff = frame_hash - last_hash
                within_force = (
                    force_interval_frames > 0
                    and (idx - last_ocr_index) < force_interval_frames
                )
                if hash_diff <= scene_threshold and within_force:
                    should_ocr = False

            if should_ocr:
                boxes = detect_boxes(
                    reader, img, scale_x, scale_y,
                    original_width, original_height, padding
                )
                if merge_pad > 0 and boxes:
                    boxes = merge_boxes(boxes, merge_pad)
                last_boxes = boxes
                last_ocr_index = idx
            else:
                boxes = last_boxes

            last_hash = frame_hash
            sample_boxes.append(boxes)

            if verbose and idx % 50 == 0:
                print(f"OCR: {idx}/{len(frame_files)} frames, boxes={len(boxes)}")

        ranges = []
        for idx, boxes in enumerate(sample_boxes):
            start_time = (idx * frame_interval) / fps
            if idx + 1 < len(sample_boxes):
                end_time = ((idx + 1) * frame_interval) / fps
            else:
                end_time = info["duration"]
            ranges.append((start_time, end_time, boxes))

        compressed = compress_ranges(ranges)
        box_ranges: list[tuple[int, int, int, int, float, float]] = []
        for start_t, end_t, boxes in compressed:
            for x, y, w, h in boxes:
                box_ranges.append((x, y, w, h, start_t, end_t))

        if not box_ranges:
            subprocess.run(["cp", input_path, output_path], check=True)
            return

        filter_str = build_ffmpeg_filter(box_ranges, max_filters=max_filters)

        if quality == "lossless":
            codec_args = ["-c:v", "libx264", "-crf", "0", "-preset", "slow"]
        elif quality == "high":
            target_bitrate = int(info["bitrate"] * 1.2)
            codec_args = ["-c:v", "libx264", "-b:v", str(target_bitrate), "-preset", "slow"]
        elif quality == "balanced":
            codec_args = ["-c:v", "libx264", "-crf", "20", "-preset", "medium"]
        else:
            codec_args = ["-c:v", "libx264", "-crf", "23", "-preset", "fast"]

        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-vf", filter_str,
            *codec_args,
            "-c:a", "copy",
            "-pix_fmt", info["pix_fmt"],
            "-movflags", "+faststart",
            output_path,
        ]
        if not verbose:
            ffmpeg_cmd.insert(1, "-loglevel")
            ffmpeg_cmd.insert(2, "warning")
        else:
            print(f"Encoding with {quality} quality, filters={len(box_ranges)}")

        result = subprocess.run(ffmpeg_cmd)
        if result.returncode != 0:
            raise RuntimeError("ffmpeg encoding failed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Block visible text in a video with low resource usage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Input video file (or playlist URL when using --playlist)",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output video file (or output directory when using --playlist/--folder)",
    )
    parser.add_argument(
        "--playlist",
        action="store_true",
        help="Treat input as a YouTube playlist URL and process all videos",
    )
    parser.add_argument(
        "--youtube",
        action="store_true",
        help="Treat input as a single YouTube video URL",
    )
    parser.add_argument(
        "--folder",
        action="store_true",
        help="Treat input as a folder of videos",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively process subfolders when using --folder",
    )
    parser.add_argument(
        "--temp-dir",
        default="temp_downloads",
        help="Temp folder for downloads when using --playlist/--youtube (default: temp_downloads)",
    )
    parser.add_argument(
        "--download-format",
        default="bestvideo[height<=720]+bestaudio/best[height<=720]",
        help="yt-dlp format string for playlist downloads",
    )
    parser.add_argument(
        "--download-merge-format",
        default="",
        help="yt-dlp merge output format (e.g., mp4). Empty = default",
    )
    parser.add_argument(
        "--keep-downloads",
        action="store_true",
        help="Keep downloaded files when using --playlist/--youtube",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip already-processed videos when using --playlist/--folder/--youtube (default: true)",
    )
    parser.add_argument(
        "--languages", "-l",
        nargs="+",
        default=["en"],
        help="Languages to detect (default: en). Use 'all' for auto-detection.",
    )
    parser.add_argument(
        "--ocr-height",
        type=int,
        default=360,
        help="Height to downscale for OCR (default: 360). Higher = slower",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=1.0,
        help="OCR sampling rate in frames/sec (default: 1.0)",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=14,
        help="Padding added around text boxes (default: 14)",
    )
    parser.add_argument(
        "--merge-pad",
        type=int,
        default=6,
        help="Merge boxes within this pixel distance (default: 6)",
    )
    parser.add_argument(
        "--scene-threshold",
        type=int,
        default=8,
        help="Scene change sensitivity for skipping OCR (default: 8)",
    )
    parser.add_argument(
        "--skip-similar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip OCR on similar frames (default: true)",
    )
    parser.add_argument(
        "--force-interval",
        type=float,
        default=2.0,
        help="Force OCR at least every N seconds (default: 2.0)",
    )
    parser.add_argument(
        "--max-filters",
        type=int,
        default=1200,
        help="Max drawbox filters before fallback (default: 1200)",
    )
    parser.add_argument(
        "-q", "--quality",
        choices=["lossless", "high", "balanced", "fast"],
        default="high",
        help="Output quality: lossless, high (default), balanced, fast",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Show progress")

    args = parser.parse_args()

    if args.playlist and args.youtube:
        print("Error: --playlist and --youtube cannot be used together", file=sys.stderr)
        sys.exit(1)
    if args.playlist and args.folder:
        print("Error: --playlist and --folder cannot be used together", file=sys.stderr)
        sys.exit(1)
    if args.youtube and args.folder:
        print("Error: --youtube and --folder cannot be used together", file=sys.stderr)
        sys.exit(1)

    mode = "file"
    if args.playlist:
        mode = "playlist"
    elif args.youtube or (args.input and is_youtube_url(args.input)):
        mode = "youtube"
    elif args.folder or (args.input and Path(args.input).exists() and Path(args.input).is_dir()):
        mode = "folder"

    if mode == "playlist":
        if not args.input:
            print("Error: Playlist URL is required when using --playlist", file=sys.stderr)
            sys.exit(1)
        output_dir = Path(args.output)
        if output_dir.exists() and not output_dir.is_dir():
            print(f"Error: Output must be a directory for playlists: {args.output}", file=sys.stderr)
            sys.exit(1)
        temp_dir = Path(args.temp_dir)
        if temp_dir.exists() and not temp_dir.is_dir():
            print(f"Error: Temp path must be a directory: {args.temp_dir}", file=sys.stderr)
            sys.exit(1)
    elif mode == "folder":
        if not args.input:
            print("Error: Input folder is required", file=sys.stderr)
            sys.exit(1)
        input_path = Path(args.input)
        if not input_path.exists() or not input_path.is_dir():
            print(f"Error: Input folder not found: {args.input}", file=sys.stderr)
            sys.exit(1)
        output_dir = Path(args.output)
        if output_dir.exists() and not output_dir.is_dir():
            print(f"Error: Output must be a directory for folders: {args.output}", file=sys.stderr)
            sys.exit(1)
    elif mode == "youtube":
        if not args.input:
            print("Error: YouTube URL is required", file=sys.stderr)
            sys.exit(1)
        temp_dir = Path(args.temp_dir)
        if temp_dir.exists() and not temp_dir.is_dir():
            print(f"Error: Temp path must be a directory: {args.temp_dir}", file=sys.stderr)
            sys.exit(1)
    else:
        if not args.input:
            print("Error: Input video file is required", file=sys.stderr)
            sys.exit(1)
        if not Path(args.input).exists():
            print(f"Error: Input file not found: {args.input}", file=sys.stderr)
            sys.exit(1)
    if args.sample_fps <= 0:
        print("Error: --sample-fps must be > 0", file=sys.stderr)
        sys.exit(1)
    if args.ocr_height <= 0:
        print("Error: --ocr-height must be > 0", file=sys.stderr)
        sys.exit(1)
    if args.force_interval < 0:
        print("Error: --force-interval must be >= 0", file=sys.stderr)
        sys.exit(1)
    if args.padding < 0 or args.merge_pad < 0:
        print("Error: --padding and --merge-pad must be >= 0", file=sys.stderr)
        sys.exit(1)

    languages = args.languages
    if "all" in languages:
        languages = ["en", "ja", "zh_sim", "ko", "es", "fr", "de", "ru", "ar", "th", "vi"]
        if args.verbose:
            print("Auto-detect mode: using multi-language model")

    try:
        if mode == "playlist":
            process_playlist(
                args.input,
                output_dir=args.output,
                temp_dir=args.temp_dir,
                download_format=args.download_format,
                merge_format=args.download_merge_format or None,
                keep_downloads=args.keep_downloads,
                skip_existing=args.skip_existing,
                languages=languages,
                ocr_height=args.ocr_height,
                sample_fps=args.sample_fps,
                padding=args.padding,
                merge_pad=args.merge_pad,
                scene_threshold=args.scene_threshold,
                skip_similar=args.skip_similar,
                force_interval=args.force_interval,
                max_filters=args.max_filters,
                quality=args.quality,
                verbose=args.verbose,
            )
        elif mode == "folder":
            process_folder(
                args.input,
                output_dir=args.output,
                recursive=args.recursive,
                skip_existing=args.skip_existing,
                languages=languages,
                ocr_height=args.ocr_height,
                sample_fps=args.sample_fps,
                padding=args.padding,
                merge_pad=args.merge_pad,
                scene_threshold=args.scene_threshold,
                skip_similar=args.skip_similar,
                force_interval=args.force_interval,
                max_filters=args.max_filters,
                quality=args.quality,
                verbose=args.verbose,
            )
        elif mode == "youtube":
            process_youtube_video(
                args.input,
                output=args.output,
                temp_dir=args.temp_dir,
                download_format=args.download_format,
                merge_format=args.download_merge_format or None,
                keep_downloads=args.keep_downloads,
                skip_existing=args.skip_existing,
                languages=languages,
                ocr_height=args.ocr_height,
                sample_fps=args.sample_fps,
                padding=args.padding,
                merge_pad=args.merge_pad,
                scene_threshold=args.scene_threshold,
                skip_similar=args.skip_similar,
                force_interval=args.force_interval,
                max_filters=args.max_filters,
                quality=args.quality,
                verbose=args.verbose,
            )
        else:
            process_video(
                args.input,
                args.output,
                languages=languages,
                ocr_height=args.ocr_height,
                sample_fps=args.sample_fps,
                padding=args.padding,
                merge_pad=args.merge_pad,
                scene_threshold=args.scene_threshold,
                skip_similar=args.skip_similar,
                force_interval=args.force_interval,
                max_filters=args.max_filters,
                quality=args.quality,
                verbose=args.verbose,
            )
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
