import struct

from bbnbp.events import ScalarEventWriter, crc32c


def test_crc32c_known_vector_and_event_framing(tmp_path):
    assert crc32c(b"123456789") == 0xe3069283
    writer = ScalarEventWriter(tmp_path)
    writer.scalar("train/loss", 12.5, 300)
    writer.close()
    data = writer.path.read_bytes()
    records = []
    while data:
        length = struct.unpack("<Q", data[:8])[0]
        payload = data[12:12+length]
        for block, checksum in ((data[:8], data[8:12]), (payload, data[12+length:16+length])):
            crc = crc32c(block)
            expected = ((crc >> 15 | crc << 17)+0xa282ead8) & 0xffffffff
            assert struct.unpack("<I", checksum)[0] == expected
        records.append(payload)
        data = data[16+length:]
    assert len(records) == 2 and b"brain.Event:2" in records[0]
    assert b"train/loss" in records[1] and b"\x15"+struct.pack("<f", 12.5) in records[1]
    assert b"\x10\xac\x02" in records[1]  # protobuf field 2, varint step=300
