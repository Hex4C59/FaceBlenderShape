"""bpy 存根：仅供 Pyright/basedpyright 解析；运行时由 Blender 提供。"""

from typing import Any, Iterator


class types:
    """Blender RNA 类型命名空间子集。"""

    class Mesh:
        """网格数据块。"""

        vertices: Any
        materials: Any
        polygons: Any
        uv_layers: Any
        users: int
        shape_keys: Any | None

    class Object:
        """场景对象。"""

        data: types.Mesh
        name: str
        type: str
        modifiers: Any

        def copy(self) -> types.Object:
            """复制对象。"""

        def update_from_editmode(self) -> None:
            """从编辑模式刷新对象数据。"""

    class Image:
        """图像数据块。"""

        size: tuple[int, int]
        channels: int
        pixels: list[float]

    class Material:
        """材质数据块。"""

        name: str
        use_nodes: bool
        node_tree: Any


class _BlendDataObjects:
    """bpy.data.objects。"""

    def get(self, key: str | None, default: Any = None) -> types.Object | None:
        """按名称查找对象；不存在则返回 default。"""

    def __getitem__(self, key: str) -> types.Object:
        """按名称取下标。"""

    def __iter__(self) -> Iterator[types.Object]:
        """遍历场景中对象。"""

    def remove(
        self,
        obj: types.Object,
        *,
        do_unlink: bool = True,
        do_id_user: bool = ...,
        do_ui_user: bool = ...,
    ) -> None:
        """移除对象数据块。"""


class _BlendDataMeshes:
    """bpy.data.meshes。"""

    def new(self, name: str) -> types.Mesh:
        """新建空网格。"""

    def remove(
        self,
        mesh: types.Mesh,
        *,
        do_unlink: bool = True,
        do_id_user: bool = ...,
        do_ui_user: bool = ...,
    ) -> None:
        """移除网格数据块。"""


class _BlendDataImages:
    """bpy.data.images。"""

    def load(self, filepath: str, *, check_existing: bool = False) -> types.Image:
        """从路径加载图像。"""


class data:
    """Blend 文件数据主入口。"""

    objects: _BlendDataObjects
    meshes: _BlendDataMeshes
    images: _BlendDataImages


class ops:
    """运算符命名空间。"""

    class object:
        """bpy.ops.object。"""

        @staticmethod
        def select_all(*, action: str) -> dict[str, Any]:
            """全选/取消全选等。"""

        @staticmethod
        def delete() -> dict[str, Any]:
            """删除选中对象。"""

    class import_scene:
        """bpy.ops.import_scene。"""

        @staticmethod
        def fbx(*, filepath: str) -> dict[str, Any]:
            """导入 FBX 文件。"""


class _ViewLayerObjects:
    """view_layer.objects。"""

    active: types.Object | None


class _ViewLayer:
    """视图层。"""

    objects: _ViewLayerObjects


class _Context:
    """当前 Blender 上下文。"""

    view_layer: _ViewLayer
    object: types.Object | None

    def evaluated_depsgraph_get(self) -> Any:
        """返回已求值的依赖图。"""


context: _Context
