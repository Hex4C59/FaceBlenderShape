"""Blender bpy API 占位存根；运行时由 Blender 提供，此处仅覆盖本文件用到的属性。"""

from typing import Any, Iterator

class _DataObjects:
    def get(self, key: str, default: Any | None = None) -> Any: ...
    def __getitem__(self, key: str) -> Any: ...

class _DataMeshes:
    def new(self, name: str) -> Any: ...
    def remove(self, mesh: Any) -> None: ...

class _DataImages:
    def __iter__(self) -> Iterator[Any]: ...
    def load(self, path: str) -> Any: ...

class _DataLights:
    def new(self, name: str, type: str) -> Any: ...

class _Data:
    objects: _DataObjects
    meshes: _DataMeshes
    images: _DataImages
    lights: _DataLights

data: _Data

class _ViewLayerObjects:
    active: Any

class _ViewLayer:
    objects: _ViewLayerObjects

class _Context:
    view_layer: _ViewLayer
    object: Any
    def evaluated_depsgraph_get(self) -> Any: ...

context: _Context

class _OpsObject:
    def select_all(self, *, action: str) -> None: ...

class _OpsImportScene:
    def fbx(self, *, filepath: str) -> None: ...

class _Ops:
    object: _OpsObject
    import_scene: _OpsImportScene

ops: _Ops
