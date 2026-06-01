"""The braille bitmap canvas: a 2x4-dots-per-cell Unicode-braille drawing
grid with one bit-plane per series. Named ``braille`` so the module alias
never collides with the ``canvas`` parameter the chart helpers pass around."""

from __future__ import annotations

BRAILLE_BASE = 0x2800
BRAILLE_BITS: dict[tuple[int, int], int] = {
    (0, 0): 0x01,
    (0, 1): 0x02,
    (0, 2): 0x04,
    (0, 3): 0x40,
    (1, 0): 0x08,
    (1, 1): 0x10,
    (1, 2): 0x20,
    (1, 3): 0x80,
}


class BrailleCanvas:
    """A multi-series braille bitmap. Each series has its own bit-plane; a cell
    is colored by the lowest-id (highest-priority) series that lit any dot in
    it."""

    def __init__(self, cols: int, rows: int, n_series: int):
        self.cols = cols
        self.rows = rows
        self.dot_w = cols * 2
        self.dot_h = rows * 4
        self._planes = [bytearray(cols * rows) for _ in range(n_series)]

    def set_dot(self, px: int, py: int, series: int) -> None:
        if 0 <= px < self.dot_w and 0 <= py < self.dot_h:
            cell = (py // 4) * self.cols + (px // 2)
            self._planes[series][cell] |= BRAILLE_BITS[(px % 2, py % 4)]

    def line(
        self, x0: int, y0: int, x1: int, y1: int, series: int, dotted: bool = False
    ) -> None:
        """Bresenham line; ``dotted`` lights every other dot (for dim series)."""
        for i, (px, py) in enumerate(_bresenham(x0, y0, x1, y1)):
            if dotted and i % 2:
                continue
            self.set_dot(px, py, series)

    def cell(self, row: int, col: int) -> tuple[str, int]:
        """Return ``(char, owner_series)`` for one cell (owner -1 if empty)."""
        bits = 0
        owner = -1
        for series, plane in enumerate(self._planes):
            value = plane[row * self.cols + col]
            if value:
                bits |= value
                if owner < 0:
                    owner = series
        return (chr(BRAILLE_BASE + bits) if bits else " "), owner


def _bresenham(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Integer line points from (x0,y0) to (x1,y1) inclusive."""
    points: list[tuple[int, int]] = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break
        err2 = 2 * err
        if err2 >= dy:
            err += dy
            x += sx
        if err2 <= dx:
            err += dx
            y += sy
    return points
