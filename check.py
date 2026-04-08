#!/usr/bin/env python3
# detect_lsb_stego.py

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from skimage.metrics import structural_similarity as ssim
except Exception:
    ssim = None


MARKER = b"<<END_OF_EXE>>"


def load_rgb(path: str) -> np.ndarray:
    """
    Load image as RGB uint8 array.
    Pillow identifies file by content, so a .jpg extension does not matter
    if the file is actually PNG data.
    """
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def save_diff_images(cover: np.ndarray, suspect: np.ndarray, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    abs_diff = np.abs(cover.astype(np.int16) - suspect.astype(np.int16)).astype(np.uint8)
    diff_gray = np.clip(np.mean(abs_diff, axis=2), 0, 255).astype(np.uint8)
    lsb_diff = ((cover & 1) ^ (suspect & 1) * 255).astype(np.uint8)

    Image.fromarray(abs_diff, mode="RGB").save(out_dir / "abs_diff_rgb.png")
    Image.fromarray(diff_gray, mode="L").save(out_dir / "abs_diff_gray.png")
    Image.fromarray(lsb_diff, mode="RGB").save(out_dir / "lsb_diff_rgb.png")


def extract_lsb_bytes(image: np.ndarray) -> bytes:
    """
    Extract all LSBs from the image and pack into bytes.
    """
    flat = image.flatten()
    bits = (flat & 1).astype(np.uint8)

    if len(bits) < 8:
        return b""

    # Trim to multiple of 8
    n = (len(bits) // 8) * 8
    bits = bits[:n]

    packed = np.packbits(bits.reshape(-1, 8), axis=1, bitorder="big")
    return packed.flatten().tobytes()


def find_marker_in_lsb_stream(image: np.ndarray, marker: bytes = MARKER) -> tuple[bool, int]:
    """
    Return (found, byte_offset).
    """
    data = extract_lsb_bytes(image)
    idx = data.find(marker)
    return (idx != -1, idx)


def compare_pair(cover: np.ndarray, suspect: np.ndarray) -> dict:
    if cover.shape != suspect.shape:
        raise ValueError(f"Image size/channel mismatch: {cover.shape} vs {suspect.shape}")

    diff = cover.astype(np.int16) - suspect.astype(np.int16)
    abs_diff = np.abs(diff)

    total = cover.size
    changed = int(np.count_nonzero(abs_diff))
    changed_rate = changed / total

    changed_lsb = int(np.count_nonzero((cover & 1) ^ (suspect & 1)))
    lsb_changed_rate = changed_lsb / total

    nonzero = abs_diff[abs_diff != 0]
    one_step_ratio = float(np.mean(nonzero == 1)) if nonzero.size else 1.0

    metrics = {
        "mse": float(np.mean((cover.astype(np.float32) - suspect.astype(np.float32)) ** 2)),
        "psnr": float(psnr(cover, suspect)),
        "changed_rate": float(changed_rate),
        "lsb_changed_rate": float(lsb_changed_rate),
        "one_step_ratio": float(one_step_ratio),
    }

    if ssim is not None:
        # SSIM expects grayscale or per-channel handling
        metrics["ssim_rgb"] = float(ssim(cover, suspect, channel_axis=2, data_range=255))
    else:
        metrics["ssim_rgb"] = None

    return metrics


def verdict(metrics: dict, marker_found: bool, marker_offset: int | None) -> str:
    """
    Simple heuristic verdict.
    """
    if marker_found:
        return f"RẤT CÓ KHẢ NĂNG LÀ ẢNH STEGO (tìm thấy marker tại byte {marker_offset})."

    # Heuristic based on LSB-only edits
    if metrics["one_step_ratio"] > 0.95 and metrics["lsb_changed_rate"] > 0.001:
        return "NGHI VẤN CAO: sai khác chủ yếu là biến đổi 1 mức và tập trung ở LSB."

    return "Không đủ dấu hiệu mạnh để kết luận bằng bộ tiêu chí hiện tại."

def extract_bit_plane(image: np.ndarray, bit: int = 0) -> np.ndarray:
    """
    Extract a specific bit-plane from an RGB image.
    bit = 0 -> LSB
    """
    return ((image >> bit) & 1).astype(np.uint8)
def save_bit_planes(cover: np.ndarray, suspect: np.ndarray, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # LSB plane (bit 0)
    cover_lsb = extract_bit_plane(cover, 0)
    suspect_lsb = extract_bit_plane(suspect, 0)

    # Scale để nhìn được (0 → 0, 1 → 255)
    cover_vis = (cover_lsb * 255).astype(np.uint8)
    suspect_vis = (suspect_lsb * 255).astype(np.uint8)

    # Difference giữa 2 LSB plane
    diff_lsb = ((cover_lsb ^ suspect_lsb) * 255).astype(np.uint8)

    # Save
    Image.fromarray(cover_vis, mode="RGB").save(out_dir / "cover_lsb.png")
    Image.fromarray(suspect_vis, mode="RGB").save(out_dir / "suspect_lsb.png")
    Image.fromarray(diff_lsb, mode="RGB").save(out_dir / "lsb_plane_diff.png")
    


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect LSB steganography from a cover image and a suspected stego image.")
    parser.add_argument("--cover", required=True, help="Original image, e.g. ptit.jpg")
    parser.add_argument("--suspect", required=True, help="Suspected stego image, e.g. hacked_ptit.jpg")
    parser.add_argument("--out", default="stego_report_out", help="Output directory for diff images")
    args = parser.parse_args()

    cover = load_rgb(args.cover)
    suspect = load_rgb(args.suspect)

    print("[*] Loaded images")
    print(f"    cover   : {args.cover} -> shape={cover.shape}")
    print(f"    suspect : {args.suspect} -> shape={suspect.shape}")

    metrics = compare_pair(cover, suspect)

    marker_found, marker_offset = find_marker_in_lsb_stream(suspect, MARKER)

    print("\n[+] Pair comparison")
    print(f"    MSE            : {metrics['mse']:.6f}")
    print(f"    PSNR           : {metrics['psnr']:.4f} dB")
    print(f"    SSIM (RGB)     : {metrics['ssim_rgb']}")
    print(f"    Changed rate   : {metrics['changed_rate'] * 100:.4f}% of all channels")
    print(f"    LSB changed    : {metrics['lsb_changed_rate'] * 100:.4f}% of all channels")
    print(f"    1-step ratio   : {metrics['one_step_ratio'] * 100:.2f}% of nonzero diffs")

    print("\n[+] Blind scan on suspect image")
    if marker_found:
        print(f"    Marker found   : YES at byte offset {marker_offset}")
    else:
        print("    Marker found   : NO")

    print("\n[=] Verdict")
    print("    " + verdict(metrics, marker_found, marker_offset))

    save_diff_images(cover, suspect, Path(args.out))
    save_bit_planes(cover, suspect, Path(args.out))
    print(f"\n[*] Saved diff images to: {args.out}")
    print("    - abs_diff_rgb.png")
    print("    - abs_diff_gray.png")
    print("    - lsb_diff_rgb.png")


if __name__ == "__main__":
    main()