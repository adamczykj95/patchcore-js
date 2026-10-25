"""Reader/writer for .pcbank memory-bank files (mirrors js/bank.js).

Layout (little endian):
  8 bytes   magic b"PCBANK1\\0"
  uint32    header length H
  H bytes   UTF-8 JSON header, space-padded so the data starts 4-byte aligned
  M*D*4     float32 bank rows
"""
import json
import struct

import numpy as np

MAGIC = b"PCBANK1\0"


def write(path, features, header):
    features = np.ascontiguousarray(features, dtype="<f4")
    header = dict(header, format=1, count=int(features.shape[0]), dim=int(features.shape[1]))
    raw = json.dumps(header).encode("utf-8")
    raw += b" " * ((-(len(MAGIC) + 4 + len(raw))) % 4)
    with open(path, "wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<I", len(raw)))
        fh.write(raw)
        fh.write(features.tobytes())


def read(path):
    with open(path, "rb") as fh:
        blob = fh.read()
    if blob[:8] != MAGIC:
        raise ValueError(f"{path} is not a .pcbank file")
    (hlen,) = struct.unpack("<I", blob[8:12])
    header = json.loads(blob[12:12 + hlen].decode("utf-8"))
    data = np.frombuffer(blob, dtype="<f4", offset=12 + hlen)
    return data.reshape(header["count"], header["dim"]), header
