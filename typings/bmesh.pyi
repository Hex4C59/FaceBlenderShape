"""bmesh 存根：仅供 Pyright/basedpyright 解析；运行时由 Blender 内置模块提供。"""

from typing import Any


class BMesh:
    """本仓库用到的 BMesh 子集。"""

    faces: Any

    def from_object(self, obj: Any, depsgraph: Any, *, cage: bool = False) -> None:
        """从已求值对象填充 BMesh。

        obj: 场景网格对象。
        depsgraph: 依赖图（evaluated depsgraph）。
        cage: 是否使用 cage 模式。
        """

    def to_mesh(self, mesh: Any) -> None:
        """将 BMesh 写入目标 Mesh 数据块。

        mesh: 目标 bpy.types.Mesh。
        """

    def free(self) -> None:
        """释放 BMesh 占用的资源。"""


class ops:
    """bmesh.ops 子模块中本仓库用到的运算。"""

    @staticmethod
    def triangulate(bm: BMesh, *, faces: Any) -> dict[str, Any]:
        """对指定面做三角化。

        bm: 目标 BMesh。
        faces: 要三角化的面序列（如 bm.faces）。
        """


def new() -> BMesh:
    """创建空 BMesh。"""

    ...
