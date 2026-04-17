#!/usr/bin/env python3
import argparse
import hashlib
import math
import struct

import numpy as np
import pywt
from PIL import Image

MAX_BYTES = 1024
DIGEST_LEN = 32
BLOCK = 8


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))


def bits_to_bytes(bits: np.ndarray) -> bytes:
    n = (len(bits) // 8) * 8
    bits = bits[:n]
    return np.packbits(bits, bitorder="big").tobytes()


def extract_coeff(coeff: float, scale: float) -> int:
    v = int(round(coeff * scale))
    return v & 1


def blind_header_check(suspect: np.ndarray, scale: float) -> tuple[bool, int]:
    _, _, b = Image.fromarray(suspect).convert("RGB").split()
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
                    bits.append(extract_coeff(LH[i, j], scale))

                    if not have_len and len(bits) >= header_target:
                        header = bits_to_bytes(np.array(bits[:header_target], dtype=np.uint8))
                        payload_len = struct.unpack(">I", header[:4])[0]
                        if payload_len > MAX_BYTES:
                            return False, -1
                        total_target = (4 + DIGEST_LEN + payload_len) * 8
                        have_len = True

                    if have_len and len(bits) >= total_target:
                        packet = bits_to_bytes(np.array(bits[:total_target], dtype=np.uint8))
                        digest = packet[4 : 4 + DIGEST_LEN]
                        payload = packet[4 + DIGEST_LEN : 4 + DIGEST_LEN + payload_len]
                        return hashlib.sha256(payload).digest() == digest, payload_len

    return False, -1


def main() -> None:
    ap = argparse.ArgumentParser(description="Block-DWT 8x8 detector demo.")
    ap.add_argument("--cover", required=True, help="Ảnh gốc")
    ap.add_argument("--suspect", required=True, help="Ảnh nghi ngờ")
    ap.add_argument("--scale", type=float, default=100.0, help="Hệ số lượng tử hóa")
    args = ap.parse_args()

    cover = np.array(Image.open(args.cover).convert("RGB"), dtype=np.uint8)
    suspect = np.array(Image.open(args.suspect).convert("RGB"), dtype=np.uint8)

    if cover.shape != suspect.shape:
        raise ValueError("Hai ảnh không cùng kích thước.")

    diff = np.abs(cover.astype(np.int16) - suspect.astype(np.int16))
    changed_rate = np.count_nonzero(diff) / diff.size

    print("[*] Pair metrics")
    print(f"  PSNR         : {psnr(cover, suspect):.4f} dB")
    print(f"  Changed rate : {changed_rate * 100:.4f}%")

    ok, payload_len = blind_header_check(suspect, args.scale)
    print("[*] Blind LH-header check")
    print(f"  Header valid : {ok}")
    print(f"  Payload len  : {payload_len if ok else 'N/A'}")

    if ok:
        print("[=] Verdict: RẤT CÓ KHẢ NĂNG là ảnh DWT-stego theo format demo.")
    elif changed_rate > 0.0005:
        print("[=] Verdict: Có sai khác nhưng chưa xác thực được header DWT.")
    else:
        print("[=] Verdict: Chưa đủ dấu hiệu mạnh.")


if __name__ == "__main__":
    main()