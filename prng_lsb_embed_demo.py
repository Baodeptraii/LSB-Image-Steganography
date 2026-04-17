#!/usr/bin/env python3
import argparse
import hashlib
import struct

import numpy as np
from PIL import Image

MAX_BYTES = 4096
DIGEST_LEN = 32  # raw SHA-256


def seed_from_key(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


def to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="big")


def build_packet(text_path: str) -> bytes:
    with open(text_path, "r", encoding="utf-8") as f:
        payload = f.read().encode("utf-8")

    if len(payload) > MAX_BYTES:
        raise ValueError(f"Payload quá lớn, tối đa {MAX_BYTES} bytes.")

    digest = hashlib.sha256(payload).digest()
    return struct.pack(">I", len(payload)) + digest + payload


def main() -> None:
    ap = argparse.ArgumentParser(description="PRNG-LSB embed demo (text-only).")
    ap.add_argument("--cover", required=True, help="Ảnh cover PNG/JPG")
    ap.add_argument("--message", required=True, help="File .txt UTF-8")
    ap.add_argument("--key", required=True, help="Khóa bí mật")
    ap.add_argument("--out", required=True, help="Ảnh stego đầu ra (PNG)")
    args = ap.parse_args()

    img = np.array(Image.open(args.cover).convert("RGB"), dtype=np.uint8)
    flat = img.reshape(-1)

    packet = build_packet(args.message)
    bits = to_bits(packet)

    if len(bits) > flat.size:
        raise ValueError("Ảnh không đủ sức chứa cho payload.")

    rng = np.random.default_rng(seed_from_key(args.key))
    perm = rng.permutation(flat.size)
    pos = perm[: len(bits)]

    flat[pos] = (flat[pos] & 0xFE) | bits
    Image.fromarray(flat.reshape(img.shape)).save(args.out, format="PNG")

    print("[OK] Nhúng PRNG-LSB xong.")
    print(f"[*] Payload bytes : {len(packet) - 4 - DIGEST_LEN}")
    print(f"[*] Bits embedded : {len(bits)}")


if __name__ == "__main__":
    main()