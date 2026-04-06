#!/usr/bin/env python3
"""Analyze LSB steganography cover/stego image pairs.

This script is tailored to the user's current implementation:
- capacity = img_array.size bits
- payload format = payload || MD5(32 ASCII chars) || end_marker
- stego image should ideally be lossless (PNG/BMP)

It computes:
- basic image metadata
- capacity / utilization / BPP
- file size delta
- pixel/element change statistics
- LSB change statistics
- MSE / RMSE / PSNR / SSIM
- grayscale histogram comparison
- grayscale entropy
- optional integrity checks if payload and extracted file are provided
- optional JSON report and histogram image
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    HAVE_SKIMAGE = True
except Exception:
    HAVE_SKIMAGE = False

try:
    import matplotlib.pyplot as plt
    HAVE_MATPLOTLIB = True
except Exception:
    HAVE_MATPLOTLIB = False


DEFAULT_END_MARKER = "<<END_OF_EXE>>"
DEFAULT_MD5_ASCII_LEN = 32


def hash_file(path: Path, algo: str = "md5") -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def image_metadata(path: Path) -> Dict[str, Any]:
    with Image.open(path) as img:
        return {
            "path": str(path),
            "filename": path.name,
            "format": img.format,
            "mode": img.mode,
            "width": img.width,
            "height": img.height,
            "file_size_bytes": path.stat().st_size,
        }


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("RGB"), dtype=np.uint8)



def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("L"), dtype=np.uint8)



def shannon_entropy(gray: np.ndarray) -> float:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    probs = hist / hist.sum()
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())



def grayscale_histogram_stats(gray_a: np.ndarray, gray_b: np.ndarray) -> Dict[str, float]:
    hist_a = np.bincount(gray_a.ravel(), minlength=256).astype(np.float64)
    hist_b = np.bincount(gray_b.ravel(), minlength=256).astype(np.float64)

    hist_a_norm = hist_a / hist_a.sum()
    hist_b_norm = hist_b / hist_b.sum()

    eps = 1e-12
    l1 = float(np.abs(hist_a_norm - hist_b_norm).sum())
    l2 = float(np.sqrt(((hist_a_norm - hist_b_norm) ** 2).sum()))
    chi_square = float(0.5 * (((hist_a_norm - hist_b_norm) ** 2) / (hist_a_norm + hist_b_norm + eps)).sum())
    intersection = float(np.minimum(hist_a_norm, hist_b_norm).sum())

    if np.std(hist_a_norm) == 0 or np.std(hist_b_norm) == 0:
        correlation = 1.0 if np.allclose(hist_a_norm, hist_b_norm) else 0.0
    else:
        correlation = float(np.corrcoef(hist_a_norm, hist_b_norm)[0, 1])

    return {
        "l1_distance": l1,
        "l2_distance": l2,
        "chi_square_distance": chi_square,
        "intersection": intersection,
        "correlation": correlation,
    }



def compare_images(cover_rgb: np.ndarray, stego_rgb: np.ndarray) -> Dict[str, Any]:
    if cover_rgb.shape != stego_rgb.shape:
        raise ValueError(
            f"Image shapes differ: cover={cover_rgb.shape}, stego={stego_rgb.shape}. "
            "Please use the true cover and the true stego image exported by your program."
        )

    cover_i = cover_rgb.astype(np.int16)
    stego_i = stego_rgb.astype(np.int16)
    diff = stego_i - cover_i
    abs_diff = np.abs(diff)

    total_elements = int(diff.size)
    total_pixels = int(diff.shape[0] * diff.shape[1])
    channels = int(diff.shape[2]) if diff.ndim == 3 else 1

    changed_elements = int(np.count_nonzero(diff))
    changed_pixels = int(np.count_nonzero(np.any(diff != 0, axis=2))) if diff.ndim == 3 else changed_elements
    lsb_changed_elements = int(np.count_nonzero((cover_rgb & 1) != (stego_rgb & 1)))

    mse = float(np.mean((stego_i - cover_i) ** 2))
    rmse = float(math.sqrt(mse))
    mae = float(np.mean(abs_diff))
    max_abs_diff = int(abs_diff.max())

    if HAVE_SKIMAGE:
        psnr = float(peak_signal_noise_ratio(cover_rgb, stego_rgb, data_range=255))
        try:
            ssim = float(structural_similarity(cover_rgb, stego_rgb, channel_axis=-1, data_range=255))
        except TypeError:
            ssim = float(structural_similarity(cover_rgb, stego_rgb, multichannel=True, data_range=255))
    else:
        if mse == 0:
            psnr = float("inf")
        else:
            psnr = float(20 * math.log10(255.0 / math.sqrt(mse)))
        ssim = None

    per_channel_changed = {}
    channel_names = ["R", "G", "B"] if channels == 3 else [f"C{i}" for i in range(channels)]
    for idx, name in enumerate(channel_names):
        per_channel_changed[name] = int(np.count_nonzero(diff[:, :, idx]))

    return {
        "shape": list(cover_rgb.shape),
        "total_pixels": total_pixels,
        "total_elements": total_elements,
        "changed_pixels": changed_pixels,
        "changed_pixels_ratio": changed_pixels / total_pixels if total_pixels else 0.0,
        "changed_elements": changed_elements,
        "changed_elements_ratio": changed_elements / total_elements if total_elements else 0.0,
        "lsb_changed_elements": lsb_changed_elements,
        "lsb_changed_ratio": lsb_changed_elements / total_elements if total_elements else 0.0,
        "per_channel_changed_elements": per_channel_changed,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "max_abs_diff": max_abs_diff,
        "psnr_db": psnr,
        "ssim": ssim,
    }



def capacity_metrics(
    cover_rgb: np.ndarray,
    payload_path: Optional[Path],
    end_marker: str,
    md5_ascii_len: int,
) -> Dict[str, Any]:
    height, width = cover_rgb.shape[:2]
    total_elements = int(cover_rgb.size)
    capacity_bits = total_elements
    capacity_bytes = capacity_bits // 8
    channels = int(cover_rgb.shape[2]) if cover_rgb.ndim == 3 else 1

    result: Dict[str, Any] = {
        "width": width,
        "height": height,
        "channels": channels,
        "capacity_bits": capacity_bits,
        "capacity_bytes": capacity_bytes,
        "capacity_kib": capacity_bytes / 1024.0,
        "max_bits_per_pixel": channels,
        "max_bytes_without_overhead": capacity_bytes,
    }

    overhead_bytes = len(end_marker.encode("utf-8")) + md5_ascii_len
    result["assumed_overhead_bytes"] = overhead_bytes
    result["max_payload_bytes_with_current_format"] = max(capacity_bytes - overhead_bytes, 0)

    if payload_path is not None and payload_path.exists():
        payload_size = payload_path.stat().st_size
        total_hidden_bytes = payload_size + overhead_bytes
        total_hidden_bits = total_hidden_bytes * 8
        result.update(
            {
                "payload_size_bytes": payload_size,
                "payload_size_kib": payload_size / 1024.0,
                "total_hidden_bytes_estimated": total_hidden_bytes,
                "total_hidden_bits_estimated": total_hidden_bits,
                "embedding_utilization_of_capacity": total_hidden_bits / capacity_bits if capacity_bits else 0.0,
                "bits_per_pixel_used": total_hidden_bits / (width * height) if width and height else 0.0,
                "bits_per_element_used": total_hidden_bits / total_elements if total_elements else 0.0,
                "fits_capacity": total_hidden_bits <= capacity_bits,
            }
        )
    else:
        result.update(
            {
                "payload_size_bytes": None,
                "payload_size_kib": None,
                "total_hidden_bytes_estimated": None,
                "total_hidden_bits_estimated": None,
                "embedding_utilization_of_capacity": None,
                "bits_per_pixel_used": None,
                "bits_per_element_used": None,
                "fits_capacity": None,
            }
        )

    return result



def integrity_metrics(payload_path: Optional[Path], extracted_path: Optional[Path]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "payload_md5": None,
        "payload_sha256": None,
        "extracted_md5": None,
        "extracted_sha256": None,
        "md5_match": None,
        "sha256_match": None,
    }

    if payload_path is not None and payload_path.exists():
        result["payload_md5"] = hash_file(payload_path, "md5")
        result["payload_sha256"] = hash_file(payload_path, "sha256")

    if extracted_path is not None and extracted_path.exists():
        result["extracted_md5"] = hash_file(extracted_path, "md5")
        result["extracted_sha256"] = hash_file(extracted_path, "sha256")

    if result["payload_md5"] and result["extracted_md5"]:
        result["md5_match"] = result["payload_md5"] == result["extracted_md5"]
    if result["payload_sha256"] and result["extracted_sha256"]:
        result["sha256_match"] = result["payload_sha256"] == result["extracted_sha256"]

    return result



def save_histogram_plot(gray_cover: np.ndarray, gray_stego: np.ndarray, out_path: Path) -> None:
    if not HAVE_MATPLOTLIB:
        raise RuntimeError("matplotlib is not installed, cannot save histogram plot.")

    hist_cover = np.bincount(gray_cover.ravel(), minlength=256)
    hist_stego = np.bincount(gray_stego.ravel(), minlength=256)

    plt.figure(figsize=(10, 5))
    plt.plot(hist_cover, label="Original Image")
    plt.plot(hist_stego, label="Stego Image")
    plt.title("Histogram comparison (Grayscale)")
    plt.xlabel("Gray level")
    plt.ylabel("Number of pixels")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()



def pretty_bytes(num_bytes: Optional[int]) -> str:
    if num_bytes is None:
        return "N/A"
    units = ["B", "KB", "MB", "GB"]
    value = float(num_bytes)
    idx = 0
    while value >= 1024 and idx < len(units) - 1:
        value /= 1024.0
        idx += 1
    return f"{value:.2f} {units[idx]}"



def build_report(args: argparse.Namespace) -> Dict[str, Any]:
    cover_path = Path(args.cover)
    stego_path = Path(args.stego)
    payload_path = Path(args.payload) if args.payload else None
    extracted_path = Path(args.extracted) if args.extracted else None

    cover_meta = image_metadata(cover_path)
    stego_meta = image_metadata(stego_path)

    cover_rgb = load_rgb(cover_path)
    stego_rgb = load_rgb(stego_path)
    cover_gray = load_gray(cover_path)
    stego_gray = load_gray(stego_path)

    report: Dict[str, Any] = {
        "cover_image": cover_meta,
        "stego_image": stego_meta,
        "file_size_delta_bytes": stego_meta["file_size_bytes"] - cover_meta["file_size_bytes"],
        "file_size_delta_percent": (
            (stego_meta["file_size_bytes"] - cover_meta["file_size_bytes"]) / cover_meta["file_size_bytes"]
            if cover_meta["file_size_bytes"] else 0.0
        ),
        "capacity": capacity_metrics(cover_rgb, payload_path, args.end_marker, args.md5_ascii_len),
        "image_difference": compare_images(cover_rgb, stego_rgb),
        "entropy": {
            "cover_gray_entropy": shannon_entropy(cover_gray),
            "stego_gray_entropy": shannon_entropy(stego_gray),
            "entropy_delta": shannon_entropy(stego_gray) - shannon_entropy(cover_gray),
        },
        "grayscale_histogram": grayscale_histogram_stats(cover_gray, stego_gray),
        "integrity": integrity_metrics(payload_path, extracted_path),
        "notes": {
            "capacity_definition": "Theo code hien tai, capacity = img_array.size bit vi moi phan tu anh mang 1 bit payload.",
            "payload_format": "payload || digest || end_marker",
            "recommended_stego_format": "PNG/BMP lossless cho LSB pixel-domain",
        },
    }

    return report



def print_summary(report: Dict[str, Any]) -> None:
    cap = report["capacity"]
    diff = report["image_difference"]
    hist = report["grayscale_histogram"]
    integrity = report["integrity"]
    cover = report["cover_image"]
    stego = report["stego_image"]

    print("=" * 72)
    print("STEGANOGRAPHY METRICS REPORT")
    print("=" * 72)
    print(f"Cover : {cover['filename']} | format={cover['format']} | mode={cover['mode']} | {cover['width']}x{cover['height']}")
    print(f"Stego : {stego['filename']} | format={stego['format']} | mode={stego['mode']} | {stego['width']}x{stego['height']}")
    print(f"Size  : {pretty_bytes(cover['file_size_bytes'])} -> {pretty_bytes(stego['file_size_bytes'])} | delta={pretty_bytes(report['file_size_delta_bytes'])}")
    print()
    print("[1] Capacity / payload")
    print(f"  Capacity bits      : {cap['capacity_bits']:,}")
    print(f"  Capacity bytes     : {cap['capacity_bytes']:,} ({cap['capacity_kib']:.2f} KiB)")
    print(f"  Max payload bytes  : {cap['max_payload_bytes_with_current_format']:,} (tru overhead MD5 + end_marker)")
    if cap['payload_size_bytes'] is not None:
        print(f"  Payload bytes      : {cap['payload_size_bytes']:,} ({cap['payload_size_kib']:.2f} KiB)")
        print(f"  Utilization        : {cap['embedding_utilization_of_capacity']:.6f}")
        print(f"  Bits per pixel     : {cap['bits_per_pixel_used']:.6f}")
        print(f"  Bits per element   : {cap['bits_per_element_used']:.6f}")
        print(f"  Fits capacity      : {cap['fits_capacity']}")
    else:
        print("  Payload bytes      : N/A (ban co the truyen --payload de tinh chinh xac)")
    print()
    print("[2] Image quality / difference")
    print(f"  Changed pixels     : {diff['changed_pixels']:,} / {diff['total_pixels']:,} ({diff['changed_pixels_ratio']:.6f})")
    print(f"  Changed elements   : {diff['changed_elements']:,} / {diff['total_elements']:,} ({diff['changed_elements_ratio']:.6f})")
    print(f"  LSB changed elems  : {diff['lsb_changed_elements']:,} / {diff['total_elements']:,} ({diff['lsb_changed_ratio']:.6f})")
    print(f"  MSE / RMSE / MAE   : {diff['mse']:.6f} / {diff['rmse']:.6f} / {diff['mae']:.6f}")
    print(f"  PSNR (dB)          : {diff['psnr_db']:.6f}")
    print(f"  SSIM               : {diff['ssim'] if diff['ssim'] is not None else 'N/A (can cai scikit-image)'}")
    print(f"  Max abs diff       : {diff['max_abs_diff']}")
    print(f"  Per-channel change : {diff['per_channel_changed_elements']}")
    print()
    print("[3] Histogram / entropy")
    print(f"  Histogram L1       : {hist['l1_distance']:.6f}")
    print(f"  Histogram L2       : {hist['l2_distance']:.6f}")
    print(f"  Chi-square         : {hist['chi_square_distance']:.6f}")
    print(f"  Intersection       : {hist['intersection']:.6f}")
    print(f"  Correlation        : {hist['correlation']:.6f}")
    print(f"  Cover entropy      : {report['entropy']['cover_gray_entropy']:.6f}")
    print(f"  Stego entropy      : {report['entropy']['stego_gray_entropy']:.6f}")
    print(f"  Entropy delta      : {report['entropy']['entropy_delta']:.6f}")
    print()
    print("[4] Integrity (optional)")
    print(f"  MD5 match          : {integrity['md5_match']}")
    print(f"  SHA-256 match      : {integrity['sha256_match']}")
    if integrity['payload_md5']:
        print(f"  Payload MD5        : {integrity['payload_md5']}")
    if integrity['extracted_md5']:
        print(f"  Extracted MD5      : {integrity['extracted_md5']}")
    print("=" * 72)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute LSB steganography metrics for a cover/stego image pair.")
    parser.add_argument("--cover", required=True, help="Path to the original cover image, e.g. ptit.jpg")
    parser.add_argument("--stego", required=True, help="Path to the stego image, e.g. hacked_ptit.jpg")
    parser.add_argument("--payload", help="Optional: original hidden file to compute utilization and hashes")
    parser.add_argument("--extracted", help="Optional: extracted file to verify integrity against payload")
    parser.add_argument("--end-marker", default=DEFAULT_END_MARKER, help="End marker used by your current implementation")
    parser.add_argument("--md5-ascii-len", type=int, default=DEFAULT_MD5_ASCII_LEN, help="ASCII length of stored MD5 digest")
    parser.add_argument("--out-json", help="Optional: save full report as JSON")
    parser.add_argument("--save-hist", help="Optional: save grayscale histogram comparison plot as PNG")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    report = build_report(args)
    print_summary(report)

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[OK] Wrote JSON report: {out_json}")

    if args.save_hist:
        gray_cover = load_gray(Path(args.cover))
        gray_stego = load_gray(Path(args.stego))
        save_histogram_plot(gray_cover, gray_stego, Path(args.save_hist))
        print(f"[OK] Wrote histogram plot: {args.save_hist}")


if __name__ == "__main__":
    main()
