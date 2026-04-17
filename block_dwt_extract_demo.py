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


def bits_to_bytes(bits: np.ndarray) -> bytes:
    n = (len(bits) // 8) * 8
    bits = bits[:n]
    return np.packbits(bits, bitorder="big").tobytes()


def extract_coeff(coeff: float, scale: float) -> int:
    v = int(round(coeff * scale))
    return v & 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Block-DWT 8x8 extract demo (text-only).")
    ap.add_argument("--stego", required=True, help="Ảnh stego")
    ap.add_argument("--out", required=True, help="File txt đầu ra")
    ap.add_argument("--scale", type=float, default=100.0, help="Hệ số lượng tử hóa")
    args = ap.parse_args()

    img = Image.open(args.stego).convert("RGB")
    _, _, b = img.split()
    bmat = np.array(b, dtype=np.float32)

    h, w = bmat.shape
    full_h = (h // BLOCK) * BLOCK
    full_w = (w // BLOCK) * BLOCK

    bits = []
    header_target = (4 + DIGEST_LEN) * 8
    total_target = header_target
    have_len = False

    for row in range(0, full_h, BLOCK):
        for col in range(0, full_w, BLOCK):
            block = bmat[row:row + BLOCK, col:col + BLOCK]
            LL, (LH, HL, HH) = pywt.dwt2(block, "haar")

            for i in range(4):
                for j in range(4):
                    bits.append(extract_coeff(LH[i, j], args.scale))

                    if not have_len and len(bits) >= header_target:
                        header = bits_to_bytes(np.array(bits[:header_target], dtype=np.uint8))
                        payload_len = struct.unpack(">I", header[:4])[0]
                        if payload_len > MAX_BYTES:
                            raise ValueError("Header không hợp lệ.")
                        total_target = (4 + DIGEST_LEN + payload_len) * 8
                        have_len = True

                    if have_len and len(bits) >= total_target:
                        packet = bits_to_bytes(np.array(bits[:total_target], dtype=np.uint8))
                        digest = packet[4 : 4 + DIGEST_LEN]
                        payload = packet[4 + DIGEST_LEN : 4 + DIGEST_LEN + payload_len]

                        if hashlib.sha256(payload).digest() != digest:
                            raise ValueError("Digest sai, dữ liệu bị hỏng.")

                        with open(args.out, "w", encoding="utf-8") as f:
                            f.write(payload.decode("utf-8"))

                        print("[OK] Tách DWT 8x8 xong.")
                        print(f"[*] Payload bytes : {payload_len}")
                        return

    raise ValueError("Không đọc đủ payload từ ảnh.")
    

if __name__ == "__main__":
    main()