"""Tests for mouse absolute coordinate conversion."""

from itacolumite.action.mouse import MOUSEEVENTF_ABSOLUTE, MOUSEEVENTF_MOVE, MOUSEEVENTF_VIRTUALDESK, _absolute_flags, _to_abs
from itacolumite.perception.display import DesktopRegion


def test_to_abs_maps_virtual_desktop_edges() -> None:
    region = DesktopRegion(left=-1920, top=0, width=3840, height=1080, target="virtual-desktop")

    assert _to_abs(-1920, 0, region) == (0, 0)
    assert _to_abs(1919, 1079, region) == (65535, 65535)


def test_absolute_flags_include_virtual_desktop_flag() -> None:
    region = DesktopRegion(left=-1920, top=0, width=3840, height=1080, target="virtual-desktop")

    flags = _absolute_flags(region, include_move=True)

    assert flags & MOUSEEVENTF_MOVE
    assert flags & MOUSEEVENTF_ABSOLUTE
    assert flags & MOUSEEVENTF_VIRTUALDESK