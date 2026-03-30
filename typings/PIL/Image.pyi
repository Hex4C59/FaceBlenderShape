"""Pillow 的 PIL.Image 子模块存根；运行时由 Pillow 提供。"""

from __future__ import annotations

class Image:
    """栅格图像实例（PIL.Image.Image）。"""

    def thumbnail(self, size: tuple[int, int], resample: int = ...) -> None:
        """按比例缩放，使图像落在 size 边界框内。"""
        ...

def fromarray(obj: object, mode: str | None = None) -> Image:
    """由 numpy 等缓冲区构造图像。"""
    ...

LANCZOS: int
