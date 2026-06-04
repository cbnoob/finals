"""Dola broadcast packet parsing (no real socket / drones needed)."""

from challenge2_swarm.dola import Dola


def _make_packet(plane_id: int, ip: str) -> bytes:
    pkt = bytearray(44)
    pkt[0] = Dola.MAVLINK_STX  # 0xFE
    pkt[5] = Dola.MSG_ID       # 232
    pkt[6:22] = bytes(range(16))            # serial number (16 bytes)
    ip_bytes = ip.encode("ascii")
    pkt[22:22 + len(ip_bytes)] = ip_bytes   # ip ascii, null padded
    pkt[38] = plane_id
    pkt[39] = 1   # wifi_mode
    pkt[40] = 0   # bind_client
    pkt[41] = 5   # wifi_power
    return bytes(pkt)


def test_parse_valid_packet():
    dola = Dola.__new__(Dola)  # skip __init__ (no socket bind)
    info = dola._parse_packet(_make_packet(3, "192.168.1.103"), "192.168.1.103")
    assert info is not None
    assert info["plane_id"] == 3
    assert info["ip"] == "192.168.1.103"
    assert info["wifi_power"] == 5


def test_reject_wrong_length():
    dola = Dola.__new__(Dola)
    assert dola._parse_packet(b"\xfe" + b"\x00" * 10, "x") is None


def test_reject_wrong_magic():
    dola = Dola.__new__(Dola)
    pkt = bytearray(_make_packet(1, "10.0.0.1"))
    pkt[0] = 0x00
    assert dola._parse_packet(bytes(pkt), "x") is None


def test_reject_wrong_msg_id():
    dola = Dola.__new__(Dola)
    pkt = bytearray(_make_packet(1, "10.0.0.1"))
    pkt[5] = 99
    assert dola._parse_packet(bytes(pkt), "x") is None
