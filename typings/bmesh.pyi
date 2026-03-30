"""Blender 内置 bmesh 的占位存根；运行时由 Blender 提供，此处仅供静态检查。"""

from typing import Any

class BMesh:
    faces: Any

    def from_object(self, *args: Any, **kwargs: Any) -> None: ...
    def to_mesh(self, mesh: Any) -> None: ...
    def free(self) -> None: ...

def new() -> BMesh: ...

class _Ops:
    @staticmethod
    def triangulate(bm: BMesh, *, faces: Any) -> None: ...

ops: _Ops
