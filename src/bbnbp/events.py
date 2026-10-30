"""Scalar TensorBoard Event/TFRecord writer using only the standard library.

Wire schema: tensorflow/core/util/event.proto and core/framework/summary.proto.
Framing: https://www.tensorflow.org/tutorials/load_data/tfrecord
This deliberately implements only version headers and legacy scalar summaries.
"""

import math
import os
import socket
import struct
import time
from pathlib import Path


def crc32c(data):
    crc = 0xffffffff
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82f63b78 if crc & 1 else 0)
    return crc ^ 0xffffffff


def _masked_crc(data):
    crc = crc32c(data)
    return ((crc >> 15 | crc << 17)+0xa282ead8) & 0xffffffff


def _varint(value):
    result = bytearray()
    while value >= 128:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def _bytes_field(number, value):
    return _varint(number*8+2)+_varint(len(value))+value


class ScalarEventWriter:
    def __init__(self, directory):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.path = path/f"events.out.tfevents.{time.time_ns()}.{socket.gethostname()}.{os.getpid()}"
        self.file = self.path.open("xb")
        self._write(b"\x09"+struct.pack("<d", time.time())+_bytes_field(3, b"brain.Event:2"))

    def _write(self, event):
        length = struct.pack("<Q", len(event))
        self.file.write(length+struct.pack("<I", _masked_crc(length))+event+struct.pack("<I", _masked_crc(event)))
        self.file.flush()

    def scalar(self, tag, value, step):
        if not math.isfinite(value) or type(step) is not int or step < 0:
            raise ValueError("Scalar events require finite values and nonnegative integer steps")
        summary_value = _bytes_field(1, tag.encode())+b"\x15"+struct.pack("<f", value)
        summary = _bytes_field(1, summary_value)
        self._write(b"\x09"+struct.pack("<d", time.time())+b"\x10"+_varint(step)+_bytes_field(5, summary))

    def close(self):
        self.file.close()
