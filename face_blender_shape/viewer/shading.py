"""Open3D 顶点色预处理与简易着色算法。"""

from __future__ import annotations

import numpy as np
import open3d as o3d

SKIN_TONE = np.array([0.80, 0.69, 0.62], dtype=np.float64)


def _vertex_neighbor_sets(vertex_count: int, faces: np.ndarray) -> list[set[int]]:
    """收集每个顶点的一环邻接顶点。

    vertex_count: 顶点总数。
    faces: 三角面索引数组，形状为 (F, 3)。
    """
    neighbors: list[set[int]] = [set() for _ in range(vertex_count)]
    for a, b, c in faces:
        ia = int(a)
        ib = int(b)
        ic = int(c)
        neighbors[ia].update((ib, ic))
        neighbors[ib].update((ia, ic))
        neighbors[ic].update((ia, ib))
    return neighbors


def _compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """根据顶点和三角面计算顶点法线。

    vertices: 顶点坐标数组，形状为 (N, 3)。
    faces: 三角面索引数组，形状为 (F, 3)。
    """
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    return np.asarray(mesh.vertex_normals, dtype=np.float64)


def _skin_detail_modulate_albedo(
    vertices: np.ndarray,
    faces: np.ndarray,
    rgb: np.ndarray,
    *,
    crease_strength: float,
    micro_strength: float,
    micro_freq: float,
) -> np.ndarray:
    """在肤色上叠加褶皱与微噪细节。

    vertices: 顶点坐标数组。
    faces: 三角面索引数组。
    rgb: 输入顶点色数组。
    crease_strength: 褶皱暗化强度。
    micro_strength: 微噪强度。
    micro_freq: 微噪空间频率。
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    out = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0).copy()
    if v.shape[0] == 0 or f.size == 0:
        return out

    normals = _compute_vertex_normals(v, f)
    neighbors = _vertex_neighbor_sets(v.shape[0], f)
    crease = np.zeros(v.shape[0], dtype=np.float64)

    for index in range(v.shape[0]):
        ring = neighbors[index]
        if not ring:
            continue
        normal = normals[index]
        acc = 0.0
        for neighbor in ring:
            acc += max(0.0, 1.0 - float(np.dot(normal, normals[neighbor])))
        crease[index] = acc / len(ring)

    q = float(np.quantile(crease, 0.92)) + 1e-6
    crease = np.clip(crease / q, 0.0, 1.0)
    out *= np.clip(1.0 - crease_strength * crease[:, np.newaxis], 0.55, 1.0)

    if micro_strength > 0.0:
        p = v * float(micro_freq)
        h = np.sin(p[:, 0]) * np.sin(p[:, 1] * 1.07) * np.sin(p[:, 2] * 0.93)
        h = (h * 0.5 + 0.5).reshape(-1, 1)
        out *= np.clip(1.0 + micro_strength * (h - 0.5) * 2.0, 0.88, 1.12)

    return np.clip(out, 0.0, 1.0)


def _matte_vertex_colors(colors: np.ndarray, matte_gamma: float) -> np.ndarray:
    """压暗过亮区域，减轻塑料感高光。

    colors: 输入顶点色数组。
    matte_gamma: 伽马值。
    """
    clipped = np.clip(np.asarray(colors, dtype=np.float64), 0.0, 1.0)
    return np.clip(np.power(clipped, float(matte_gamma)), 0.0, 1.0)


def _light_dir_for_mesh(vertices: np.ndarray) -> np.ndarray:
    """根据网格朝向估计一个简化主光方向。

    vertices: 顶点坐标数组。
    """
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    if float(extent[2]) > float(extent[1]):
        direction = np.array([0.28, 0.62, 0.74], dtype=np.float64)
    else:
        direction = np.array([0.38, 0.72, 0.58], dtype=np.float64)
    direction /= np.linalg.norm(direction) + 1e-9
    return direction


def _bake_half_lambert_shading(
    vertices: np.ndarray,
    faces: np.ndarray,
    albedo: np.ndarray,
    *,
    baked_ambient: float,
    baked_diffuse: float,
) -> np.ndarray:
    """把半 Lambert 漫反射烘焙进顶点色。

    vertices: 顶点坐标数组。
    faces: 三角面索引数组。
    albedo: 输入反照率数组。
    baked_ambient: 环境光系数。
    baked_diffuse: 漫反射系数。
    """
    normals = _compute_vertex_normals(vertices, faces)
    light_dir = _light_dir_for_mesh(vertices)
    ndotl = (normals * light_dir).sum(axis=1, keepdims=True)
    half_lambert = np.clip(0.5 * ndotl + 0.5, 0.0, 1.0)
    lit = float(baked_ambient) + float(baked_diffuse) * half_lambert
    rgb = np.clip(np.asarray(albedo, dtype=np.float64), 0.0, 1.0) * lit
    shadow = np.clip(1.0 - lit, 0.0, 1.0)
    warm = np.concatenate([0.04 * shadow, 0.015 * shadow, 0.012 * shadow], axis=1)
    return np.clip(rgb + warm, 0.0, 1.0)


def prepare_display_colors(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    vertex_colors: np.ndarray | None,
    skin_detail_enabled: bool,
    skin_crease_strength: float,
    skin_micro_strength: float,
    skin_micro_freq: float,
    matte_gamma: float,
    baked_shading: bool,
    baked_ambient: float,
    baked_diffuse: float,
) -> np.ndarray:
    """按 viewer 配置生成最终显示用顶点色。

    vertices: 顶点坐标数组。
    faces: 三角面索引数组。
    vertex_colors: 外部传入的原始顶点色；None 时使用默认肤色。
    skin_detail_enabled: 是否启用皮肤细节。
    skin_crease_strength: 褶皱强度。
    skin_micro_strength: 微噪强度。
    skin_micro_freq: 微噪频率。
    matte_gamma: 亮部压暗系数。
    baked_shading: 是否启用烘焙光照。
    baked_ambient: 环境光系数。
    baked_diffuse: 漫反射系数。
    """
    if vertex_colors is None:
        colors = np.tile(SKIN_TONE, (len(vertices), 1))
    else:
        colors = np.clip(np.asarray(vertex_colors, dtype=np.float64), 0.0, 1.0)

    has_mesh = len(vertices) > 0 and len(faces) > 0
    has_skin_detail = skin_crease_strength > 0.0 or skin_micro_strength > 0.0

    if skin_detail_enabled and has_mesh and has_skin_detail:
        colors = _skin_detail_modulate_albedo(
            vertices,
            faces,
            colors,
            crease_strength=skin_crease_strength,
            micro_strength=skin_micro_strength,
            micro_freq=skin_micro_freq,
        )

    colors = _matte_vertex_colors(colors, matte_gamma)

    if baked_shading and has_mesh:
        colors = _bake_half_lambert_shading(
            vertices,
            faces,
            colors,
            baked_ambient=baked_ambient,
            baked_diffuse=baked_diffuse,
        )

    return colors
