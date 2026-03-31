"""open3d.geometry 占位存根。"""

from typing import Any

class LineSet:
    points: Any
    lines: Any
    colors: Any
    def __init__(self) -> None: ...

class TriangleMesh:
    vertices: Any
    triangles: Any
    vertex_colors: Any
    def __init__(self) -> None: ...
    def compute_vertex_normals(self) -> None: ...
