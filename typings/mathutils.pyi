"""mathutils 存根：仅供 Pyright/basedpyright 解析；运行时由 Blender 内置模块提供。"""

class Vector:
    """Blender 数学向量；此处仅声明本仓库用到的接口。"""

    x: float
    y: float
    z: float

    def __init__(self, *args: object) -> None: ...
    def __sub__(self, other: Vector) -> Vector: ...
    def cross(self, other: Vector) -> Vector: ...
    @property
    def length(self) -> float: ...
