#!/usr/bin/env python3
import argparse
import hashlib
import struct

import numpy as np
import pywt
from PIL import Image

MAX_BYTES = 1024
DIGEST_LEN = 32
BLOCK = 8


def to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="big")


def build_packet(text_path: str) -> bytes:
    with open(text_path, "r", encoding="utf-8") as f:
        payload = f.read().encode("utf-8")

    if len(payload) > MAX_BYTES:
        raise ValueError(f"Payload quá lớn, tối đa {MAX_BYTES} bytes.")

    digest = hashlib.sha256(payload).digest()
    return struct.pack(">I", len(payload)) + digest + payload


def embed_coeff(coeff: float, bit: int, scale: float) -> float:
    v = int(round(coeff * scale))
    v = (v & ~1) | int(bit)
    return float(v) / scale


def main() -> None:
    ap = argparse.ArgumentParser(description="Block-DWT 8x8 embed demo (text-only).")
    ap.add_argument("--cover", required=True, help="Ảnh cover")
    ap.add_argument("--message", required=True, help="File txt UTF-8")
    ap.add_argument("--out", required=True, help="Ảnh stego PNG")
    ap.add_argument("--scale", type=float, default=100.0, help="Hệ số lượng tử hóa")
    args = ap.parse_args()

    img = Image.open(args.cover).convert("RGB")
    r, g, b = img.split()
    bmat = np.array(b, dtype=np.float32)

    packet = build_packet(args.message)
    bits = to_bits(packet)
    bit_idx = 0

    h, w = bmat.shape
    full_h = (h // BLOCK) * BLOCK
    full_w = (w // BLOCK) * BLOCK
    capacity_bits = (full_h // BLOCK) * (full_w // BLOCK) * 16
    if len(bits) > capacity_bits:
        raise ValueError("Ảnh không đủ sức chứa theo cấu hình DWT hiện tại.")

    for row in range(0, full_h, BLOCK):
        for col in range(0, full_w, BLOCK):
            if bit_idx >= len(bits):
                break

            block = bmat[row:row + BLOCK, col:col + BLOCK]
            LL, (LH, HL, HH) = pywt.dwt2(block, "haar")

            for i in range(4):
                for j in range(4):
                    if bit_idx < len(bits):
                        LH[i, j] = embed_coeff(LH[i, j], int(bits[bit_idx]), args.scale)
                        bit_idx += 1

            bmat[row:row + BLOCK, col:col + BLOCK] = pywt.idwt2((LL, (LH, HL, HH)), "haar")

    if bit_idx < len(bits):
        raise ValueError("Nhúng chưa đủ toàn bộ payload.")

    b_stego = Image.fromarray(np.clip(bmat, 0, 255).astype(np.uint8))
    Image.merge("RGB", (r, g, b_stego)).save(args.out, format="PNG")

    print("[OK] Nhúng DWT 8x8 xong.")
    print(f"[*] Payload bytes : {len(packet) - 4 - DIGEST_LEN}")
    print(f"[*] Bits embedded : {len(bits)}")
    print(f"[*] Capacity bits : {capacity_bits}")


if __name__ == "__main__":
    main()