#!/usr/bin/env python3
import argparse
import hashlib
import math
import struct

import numpy as np
from PIL import Image

MAX_BYTES = 4096
DIGEST_LEN = 32


def seed_from_key(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


def bits_to_bytes(bits: np.ndarray) -> bytes:
    n = (len(bits) // 8) * 8
    bits = bits[:n]
    return np.packbits(bits, bitorder="big").tobytes()


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))


def main() -> None:
    ap = argparse.ArgumentParser(description="PRNG-LSB detector demo.")
    ap.add_argument("--cover", required=True, help="Ảnh gốc")
    ap.add_argument("--suspect", required=True, help="Ảnh nghi ngờ")
    ap.add_argument("--key", help="Khóa để thử xác thực header")
    args = ap.parse_args()

    cover = np.array(Image.open(args.cover).convert("RGB"), dtype=np.uint8)
    suspect = np.array(Image.open(args.suspect).convert("RGB"), dtype=np.uint8)

    if cover.shape != suspect.shape:
        raise ValueError("Hai ảnh không cùng kích thước.")

    diff = np.abs(cover.astype(np.int16) - suspect.astype(np.int16))
    changed_rate = np.count_nonzero(diff) / diff.size
    lsb_changed_rate = np.count_nonzero((cover & 1) ^ (suspect & 1)) / diff.size
    nz = diff[diff != 0]
    one_step_ratio = float(np.mean(nz == 1)) if nz.size else 1.0

    print("[*] Pair metrics")
    print(f"  PSNR            : {psnr(cover, suspect):.4f} dB")
    print(f"  Changed rate    : {changed_rate * 100:.4f}%")
    print(f"  LSB changed     : {lsb_changed_rate * 100:.4f}%")
    print(f"  1-step ratio    : {one_step_ratio * 100:.2f}%")

    if args.key:
        flat = suspect.reshape(-1)
        rng = np.random.default_rng(seed_from_key(args.key))
        perm = rng.permutation(flat.size)

        try:
            header_bits = (4 + DIGEST_LEN) * 8
            header = bits_to_bytes(flat[perm[:header_bits]] & 1)
            payload_len = struct.unpack(">I", header[:4])[0]

            if payload_len > MAX_BYTES:
                raise ValueError

            total_bits = (4 + DIGEST_LEN + payload_len) * 8
            packet = bits_to_bytes(flat[perm[:total_bits]] & 1)
            digest = packet[4 : 4 + DIGEST_LEN]
            payload = packet[4 + DIGEST_LEN : 4 + DIGEST_LEN + payload_len]
            ok = hashlib.sha256(payload).digest() == digest

            print("[*] Keyed check")
            print(f"  Header length   : {payload_len}")
            print(f"  Digest valid    : {ok}")
        except Exception:
            print("[*] Keyed check")
            print("  Không xác thực được header/digest.")

    if one_step_ratio > 0.95 and lsb_changed_rate > 0.001:
        print("[=] Verdict: NGHI VẤN CAO (mẫu sửa 1 mức và tập trung ở LSB).")
    else:
        print("[=] Verdict: Chưa đủ dấu hiệu mạnh với bộ tiêu chí hiện tại.")


if __name__ == "__main__":
    main()