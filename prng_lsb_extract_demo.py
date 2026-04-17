#!/usr/bin/env python3
import argparse
import hashlib
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


def main() -> None:
    ap = argparse.ArgumentParser(description="PRNG-LSB extract demo (text-only).")
    ap.add_argument("--stego", required=True, help="Ảnh stego PNG")
    ap.add_argument("--key", required=True, help="Khóa bí mật")
    ap.add_argument("--out", required=True, help="File txt đầu ra")
    args = ap.parse_args()

    img = np.array(Image.open(args.stego).convert("RGB"), dtype=np.uint8)
    flat = img.reshape(-1)

    rng = np.random.default_rng(seed_from_key(args.key))
    perm = rng.permutation(flat.size)

    header_bits = (4 + DIGEST_LEN) * 8
    header = bits_to_bytes(flat[perm[:header_bits]] & 1)

    payload_len = struct.unpack(">I", header[:4])[0]
    if payload_len > MAX_BYTES:
        raise ValueError("Header không hợp lệ hoặc sai khóa.")

    total_bits = (4 + DIGEST_LEN + payload_len) * 8
    packet = bits_to_bytes(flat[perm[:total_bits]] & 1)

    digest = packet[4 : 4 + DIGEST_LEN]
    payload = packet[4 + DIGEST_LEN : 4 + DIGEST_LEN + payload_len]

    if hashlib.sha256(payload).digest() != digest:
        raise ValueError("Sai khóa hoặc dữ liệu đã bị hỏng.")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(payload.decode("utf-8"))

    print("[OK] Tách PRNG-LSB xong.")
    print(f"[*] Payload bytes : {payload_len}")


if __name__ == "__main__":
    main()