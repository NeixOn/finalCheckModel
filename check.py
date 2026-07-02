# evaluate_single_3d.py

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from PIL import Image


# -----------------------------
# Загрузка и базовая обработка
# -----------------------------

def load_mesh(path: str) -> trimesh.Trimesh:
    """
    Загружает mesh из obj/ply/glb/stl и приводит Scene к одному Trimesh.
    """
    mesh_or_scene = trimesh.load(path, force="scene")

    if isinstance(mesh_or_scene, trimesh.Scene):
        geometries = []
        for geom in mesh_or_scene.geometry.values():
            if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) > 0:
                geometries.append(geom)
        if not geometries:
            raise ValueError(f"В файле нет валидной геометрии: {path}")
        mesh = trimesh.util.concatenate(geometries)
    elif isinstance(mesh_or_scene, trimesh.Trimesh):
        mesh = mesh_or_scene
    else:
        raise ValueError(f"Не удалось загрузить mesh: {path}")

    mesh.remove_unreferenced_vertices()
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    mesh.process(validate=False)

    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"Пустой mesh: {path}")

    return mesh


def copy_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    return trimesh.Trimesh(
        vertices=np.array(mesh.vertices, dtype=np.float64),
        faces=np.array(mesh.faces, dtype=np.int64),
        process=False
    )


def normalize_mesh_independent(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Нормализует каждую модель независимо:
    центр bbox -> 0, максимальная сторона bbox -> 1.

    Это удобно, если сгенерированная и исходная модели находятся
    в разных системах координат.
    """
    mesh = copy_mesh(mesh)
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2.0
    extents = bounds[1] - bounds[0]
    scale = float(np.max(extents))

    if scale <= 1e-12:
        raise ValueError("Невозможно нормализовать mesh: нулевой размер bbox")

    mesh.vertices = (mesh.vertices - center) / scale
    return mesh


def normalize_mesh_by_gt(pred: trimesh.Trimesh, gt: trimesh.Trimesh):
    """
    Нормализует обе модели по bbox исходной модели.
    Это сохраняет ошибку масштаба/смещения предсказания,
    но требует, чтобы обе модели были в одной системе координат.
    """
    pred = copy_mesh(pred)
    gt = copy_mesh(gt)

    bounds = gt.bounds
    center = (bounds[0] + bounds[1]) / 2.0
    extents = bounds[1] - bounds[0]
    scale = float(np.max(extents))

    if scale <= 1e-12:
        raise ValueError("Невозможно нормализовать по GT: нулевой размер bbox")

    pred.vertices = (pred.vertices - center) / scale
    gt.vertices = (gt.vertices - center) / scale

    return pred, gt


def normalize_pair(pred: trimesh.Trimesh, gt: trimesh.Trimesh, mode: str):
    """
    mode:
    - independent: каждую модель центрируем и масштабируем отдельно.
      Хорошо для сравнения формы.
    - gt: обе модели нормализуем по исходной модели.
      Хорошо, если координаты уже согласованы.
    - none: ничего не делаем.
    """
    if mode == "independent":
        return normalize_mesh_independent(pred), normalize_mesh_independent(gt)
    if mode == "gt":
        return normalize_mesh_by_gt(pred, gt)
    if mode == "none":
        return copy_mesh(pred), copy_mesh(gt)

    raise ValueError(f"Неизвестный режим нормализации: {mode}")


# -----------------------------
# Информация о mesh
# -----------------------------

def mesh_basic_stats(mesh: trimesh.Trimesh) -> dict:
    bounds = mesh.bounds
    extents = bounds[1] - bounds[0]

    face_areas = mesh.area_faces if len(mesh.faces) > 0 else np.array([])
    degenerate_faces = int(np.sum(face_areas < 1e-12)) if len(face_areas) else 0

    # Длины ребер
    edges = mesh.edges_unique
    if len(edges) > 0:
        edge_lengths = np.linalg.norm(
            mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]],
            axis=1
        )
        edge_stats = {
            "edge_length_min": float(np.min(edge_lengths)),
            "edge_length_mean": float(np.mean(edge_lengths)),
            "edge_length_std": float(np.std(edge_lengths)),
            "edge_length_max": float(np.max(edge_lengths)),
        }
    else:
        edge_stats = {
            "edge_length_min": None,
            "edge_length_mean": None,
            "edge_length_std": None,
            "edge_length_max": None,
        }

    # Boundary / non-manifold edges
    try:
        inv = mesh.edges_unique_inverse
        counts = np.bincount(inv)
        boundary_edges = int(np.sum(counts == 1))
        nonmanifold_edges = int(np.sum(counts > 2))
    except Exception:
        boundary_edges = None
        nonmanifold_edges = None

    # Компоненты связности
    try:
        components = mesh.split(only_watertight=False)
        component_count = int(len(components))
        largest_component_faces = int(max(len(c.faces) for c in components)) if components else 0
    except Exception:
        component_count = None
        largest_component_faces = None

    volume = None
    abs_volume = None
    if mesh.is_watertight:
        try:
            volume = float(mesh.volume)
            abs_volume = float(abs(mesh.volume))
        except Exception:
            pass

    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "surface_area": float(mesh.area),
        "volume_signed_if_watertight": volume,
        "volume_abs_if_watertight": abs_volume,
        "is_watertight": bool(mesh.is_watertight),
        "is_winding_consistent": bool(mesh.is_winding_consistent),
        "euler_number": int(mesh.euler_number),
        "component_count": component_count,
        "largest_component_faces": largest_component_faces,
        "degenerate_faces": degenerate_faces,
        "boundary_edges": boundary_edges,
        "nonmanifold_edges": nonmanifold_edges,
        "bbox_min": bounds[0].tolist(),
        "bbox_max": bounds[1].tolist(),
        "bbox_extents": extents.tolist(),
        "bbox_diagonal": float(np.linalg.norm(extents)),
        **edge_stats
    }


def compare_basic_stats(pred: trimesh.Trimesh, gt: trimesh.Trimesh) -> dict:
    pred_ext = pred.bounds[1] - pred.bounds[0]
    gt_ext = gt.bounds[1] - gt.bounds[0]

    def safe_ratio(a, b):
        return float(a / b) if abs(b) > 1e-12 else None

    pred_bbox_volume = float(np.prod(pred_ext))
    gt_bbox_volume = float(np.prod(gt_ext))

    area_ratio = safe_ratio(pred.area, gt.area)
    bbox_volume_ratio = safe_ratio(pred_bbox_volume, gt_bbox_volume)

    volume_ratio = None
    if pred.is_watertight and gt.is_watertight:
        if abs(gt.volume) > 1e-12:
            volume_ratio = float(abs(pred.volume) / abs(gt.volume))

    return {
        "surface_area_ratio_pred_to_gt": area_ratio,
        "bbox_volume_ratio_pred_to_gt": bbox_volume_ratio,
        "mesh_volume_ratio_pred_to_gt_if_watertight": volume_ratio,
        "bbox_extents_abs_error": np.abs(pred_ext - gt_ext).tolist(),
        "bbox_extents_relative_error": [
            safe_ratio(abs(pred_ext[i] - gt_ext[i]), gt_ext[i])
            for i in range(3)
        ],
    }


# -----------------------------
# Сэмплирование поверхности
# -----------------------------

def sample_surface_with_normals(mesh: trimesh.Trimesh, n: int, seed: int):
    """
    Сэмплирует точки на поверхности и возвращает нормали соответствующих граней.
    """
    rng_state = np.random.get_state()
    np.random.seed(seed)

    points, face_ids = trimesh.sample.sample_surface(mesh, n)

    np.random.set_state(rng_state)

    normals = mesh.face_normals[face_ids]
    return points.astype(np.float64), normals.astype(np.float64)


# -----------------------------
# Геометрические метрики
# -----------------------------

def distance_metrics(pred: trimesh.Trimesh, gt: trimesh.Trimesh, samples: int, seed: int, thresholds):
    pred_pts, pred_normals = sample_surface_with_normals(pred, samples, seed)
    gt_pts, gt_normals = sample_surface_with_normals(gt, samples, seed + 1)

    gt_tree = cKDTree(gt_pts)
    pred_tree = cKDTree(pred_pts)

    d_pred_to_gt, idx_pred_to_gt = gt_tree.query(pred_pts, k=1)
    d_gt_to_pred, idx_gt_to_pred = pred_tree.query(gt_pts, k=1)

    chamfer_l1 = float(np.mean(d_pred_to_gt) + np.mean(d_gt_to_pred))
    chamfer_l2 = float(np.mean(d_pred_to_gt ** 2) + np.mean(d_gt_to_pred ** 2))

    hausdorff = float(max(np.max(d_pred_to_gt), np.max(d_gt_to_pred)))

    rms_pred_to_gt = float(np.sqrt(np.mean(d_pred_to_gt ** 2)))
    rms_gt_to_pred = float(np.sqrt(np.mean(d_gt_to_pred ** 2)))
    rms_symmetric = float((rms_pred_to_gt + rms_gt_to_pred) / 2.0)

    # Normal consistency.
    # abs=True означает, что перевернутые нормали не штрафуются так сильно.
    nearest_gt_normals = gt_normals[idx_pred_to_gt]
    nearest_pred_normals = pred_normals[idx_gt_to_pred]

    normal_dot_pred_to_gt = np.sum(pred_normals * nearest_gt_normals, axis=1)
    normal_dot_gt_to_pred = np.sum(gt_normals * nearest_pred_normals, axis=1)

    normal_consistency_abs = float(
        0.5 * (
            np.mean(np.abs(normal_dot_pred_to_gt)) +
            np.mean(np.abs(normal_dot_gt_to_pred))
        )
    )

    normal_consistency_oriented = float(
        0.5 * (
            np.mean(normal_dot_pred_to_gt) +
            np.mean(normal_dot_gt_to_pred)
        )
    )

    f_scores = {}
    for t in thresholds:
        precision = float(np.mean(d_pred_to_gt <= t))
        recall = float(np.mean(d_gt_to_pred <= t))
        if precision + recall > 1e-12:
            f = float(2.0 * precision * recall / (precision + recall))
        else:
            f = 0.0

        f_scores[f"threshold_{t:g}"] = {
            "precision": precision,
            "recall": recall,
            "f_score": f
        }

    return {
        "samples_per_mesh": int(samples),

        "chamfer_l1_mean_sum": chamfer_l1,
        "chamfer_l2_mean_squared_sum": chamfer_l2,

        "hausdorff_approx": hausdorff,

        "pred_to_gt_mean": float(np.mean(d_pred_to_gt)),
        "pred_to_gt_median": float(np.median(d_pred_to_gt)),
        "pred_to_gt_rms": rms_pred_to_gt,
        "pred_to_gt_p90": float(np.percentile(d_pred_to_gt, 90)),
        "pred_to_gt_p95": float(np.percentile(d_pred_to_gt, 95)),
        "pred_to_gt_p99": float(np.percentile(d_pred_to_gt, 99)),
        "pred_to_gt_max": float(np.max(d_pred_to_gt)),

        "gt_to_pred_mean": float(np.mean(d_gt_to_pred)),
        "gt_to_pred_median": float(np.median(d_gt_to_pred)),
        "gt_to_pred_rms": rms_gt_to_pred,
        "gt_to_pred_p90": float(np.percentile(d_gt_to_pred, 90)),
        "gt_to_pred_p95": float(np.percentile(d_gt_to_pred, 95)),
        "gt_to_pred_p99": float(np.percentile(d_gt_to_pred, 99)),
        "gt_to_pred_max": float(np.max(d_gt_to_pred)),

        "rms_symmetric": rms_symmetric,

        "normal_consistency_abs": normal_consistency_abs,
        "normal_consistency_oriented": normal_consistency_oriented,

        "f_scores": f_scores,

        # Для сохранения гистограмм.
        "_raw_distances": {
            "pred_to_gt": d_pred_to_gt,
            "gt_to_pred": d_gt_to_pred,
        }
    }


# -----------------------------
# Voxel IoU
# -----------------------------

def voxel_iou_contains(pred: trimesh.Trimesh, gt: trimesh.Trimesh, resolution: int = 64) -> dict:
    """
    Примерная объемная IoU.
    Работает корректнее для watertight mesh.
    Использует mesh.contains, поэтому может требовать rtree.
    """
    if not pred.is_watertight or not gt.is_watertight:
        return {
            "voxel_iou": None,
            "voxel_resolution": resolution,
            "note": "Voxel IoU не рассчитан: pred или gt не является watertight."
        }

    try:
        # Общая область проверки
        all_min = np.minimum(pred.bounds[0], gt.bounds[0])
        all_max = np.maximum(pred.bounds[1], gt.bounds[1])

        # Небольшой отступ
        pad = 0.02 * np.max(all_max - all_min)
        all_min = all_min - pad
        all_max = all_max + pad

        xs = np.linspace(all_min[0], all_max[0], resolution)
        ys = np.linspace(all_min[1], all_max[1], resolution)
        zs = np.linspace(all_min[2], all_max[2], resolution)

        grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1)
        points = grid.reshape(-1, 3)

        pred_inside = pred.contains(points)
        gt_inside = gt.contains(points)

        intersection = np.logical_and(pred_inside, gt_inside).sum()
        union = np.logical_or(pred_inside, gt_inside).sum()

        iou = float(intersection / union) if union > 0 else None

        return {
            "voxel_iou": iou,
            "voxel_resolution": resolution,
            "intersection_voxels": int(intersection),
            "union_voxels": int(union),
            "note": "Voxel IoU рассчитан по регулярной сетке точек."
        }

    except Exception as e:
        return {
            "voxel_iou": None,
            "voxel_resolution": resolution,
            "note": f"Voxel IoU не рассчитан из-за ошибки: {repr(e)}"
        }


# -----------------------------
# Анализ изображения
# -----------------------------

def analyze_image(image_path: str) -> dict:
    """
    Фото само по себе не дает 3D-метрик без камеры.
    Здесь считаем только базовую информацию и примерную долю объекта.
    """
    img = Image.open(image_path).convert("RGBA")
    arr = np.asarray(img).astype(np.uint8)

    h, w = arr.shape[:2]
    alpha = arr[:, :, 3]

    # Если есть альфа-канал, используем его.
    alpha_foreground = alpha > 10
    alpha_ratio = float(alpha_foreground.mean())

    # Простая эвристика для белого фона:
    # считаем foreground пикселями те, которые не почти белые.
    rgb = arr[:, :, :3].astype(np.int16)
    non_white = np.any(rgb < 245, axis=2)
    non_white_ratio = float(non_white.mean())

    return {
        "image_path": str(image_path),
        "width": int(w),
        "height": int(h),
        "mode_after_load": "RGBA",
        "foreground_ratio_by_alpha_if_available": alpha_ratio,
        "foreground_ratio_by_non_white_heuristic": non_white_ratio,
        "note": (
            "Фото используется как входное изображение. "
            "Строгая silhouette-метрика не считается, потому что для нее нужны параметры камеры "
            "или тот же ракурс рендера модели."
        )
    }


# -----------------------------
# Сохранение гистограмм
# -----------------------------

def save_distance_histograms(raw_distances: dict, out_dir: str):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    for name, values in raw_distances.items():
        plt.figure(figsize=(8, 5))
        plt.hist(values, bins=80)
        plt.xlabel("Distance")
        plt.ylabel("Count")
        plt.title(name)
        plt.tight_layout()

        out_path = out_dir / f"{name}_distance_hist.png"
        plt.savefig(out_path, dpi=160)
        plt.close()

        paths[name] = str(out_path)

    return paths


# -----------------------------
# Текстовая интерпретация
# -----------------------------

def interpret_report(report: dict) -> list:
    lines = []

    dm = report["geometry_metrics"]

    chamfer = dm["chamfer_l1_mean_sum"]
    haus = dm["hausdorff_approx"]
    normal_abs = dm["normal_consistency_abs"]

    lines.append("Краткая интерпретация:")

    if chamfer < 0.03:
        lines.append("- Chamfer Distance низкий: форма близка к эталону.")
    elif chamfer < 0.08:
        lines.append("- Chamfer Distance средний: общая форма похожа, но есть заметные отклонения.")
    else:
        lines.append("- Chamfer Distance высокий: геометрия заметно отличается от эталона.")

    if haus < 0.1:
        lines.append("- Hausdorff Distance невысокий: грубых дальних выбросов немного.")
    else:
        lines.append("- Hausdorff Distance высокий: возможны выбросы, лишние части или сильные локальные ошибки.")

    if normal_abs > 0.85:
        lines.append("- Нормали хорошо согласованы: локальная ориентация поверхностей похожа.")
    elif normal_abs > 0.65:
        lines.append("- Нормали согласованы умеренно: поверхность частично похожа, но есть сглаживание или артефакты.")
    else:
        lines.append("- Нормали согласованы слабо: локальная поверхность сильно отличается или нормали нестабильны.")

    pred_stats = report["mesh_quality"]["pred"]
    if not pred_stats["is_watertight"]:
        lines.append("- Сгенерированная модель не watertight: объемные метрики и 3D-печать могут быть проблемными.")

    if pred_stats["boundary_edges"] is not None and pred_stats["boundary_edges"] > 0:
        lines.append(f"- Найдены boundary edges: {pred_stats['boundary_edges']}. Это признак отверстий или открытой поверхности.")

    if pred_stats["nonmanifold_edges"] is not None and pred_stats["nonmanifold_edges"] > 0:
        lines.append(f"- Найдены non-manifold edges: {pred_stats['nonmanifold_edges']}. Это топологические дефекты mesh.")

    return lines


# -----------------------------
# Главная функция
# -----------------------------

def evaluate_one(
    pred_path: str,
    gt_path: str,
    image_path: str,
    out_dir: str,
    samples: int,
    seed: int,
    normalization: str,
    voxel_resolution: int,
    thresholds
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_raw = load_mesh(pred_path)
    gt_raw = load_mesh(gt_path)

    pred_norm, gt_norm = normalize_pair(pred_raw, gt_raw, normalization)

    pred_stats_raw = mesh_basic_stats(pred_raw)
    gt_stats_raw = mesh_basic_stats(gt_raw)

    pred_stats_norm = mesh_basic_stats(pred_norm)
    gt_stats_norm = mesh_basic_stats(gt_norm)

    stats_compare_raw = compare_basic_stats(pred_raw, gt_raw)
    stats_compare_norm = compare_basic_stats(pred_norm, gt_norm)

    geom = distance_metrics(
        pred=pred_norm,
        gt=gt_norm,
        samples=samples,
        seed=seed,
        thresholds=thresholds
    )

    raw_distances = geom.pop("_raw_distances")
    hist_paths = save_distance_histograms(raw_distances, out_dir)

    voxel = voxel_iou_contains(pred_norm, gt_norm, resolution=voxel_resolution)

    image_info = analyze_image(image_path)

    report = {
        "input": {
            "pred_model": str(pred_path),
            "gt_model": str(gt_path),
            "image": str(image_path),
            "samples": int(samples),
            "seed": int(seed),
            "normalization": normalization,
            "thresholds_for_f_score": thresholds,
        },
        "image_info": image_info,
        "mesh_quality": {
            "pred": pred_stats_raw,
            "gt": gt_stats_raw,
            "pred_normalized": pred_stats_norm,
            "gt_normalized": gt_stats_norm,
        },
        "mesh_comparison": {
            "raw": stats_compare_raw,
            "normalized": stats_compare_norm,
        },
        "geometry_metrics": geom,
        "voxel_metrics": voxel,
        "saved_files": {
            "distance_histograms": hist_paths
        }
    }

    report["interpretation"] = interpret_report(report)

    json_path = out_dir / "evaluation_report.json"
    txt_path = out_dir / "evaluation_report.txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(format_report_text(report))

    print(format_report_text(report))
    print(f"\nJSON отчет сохранен: {json_path}")
    print(f"TXT отчет сохранен:  {txt_path}")

    return report


def format_report_text(report: dict) -> str:
    lines = []

    lines.append("=" * 80)
    lines.append("ОЦЕНКА КАЧЕСТВА 3D-РЕКОНСТРУКЦИИ")
    lines.append("=" * 80)

    lines.append("\n[Входные данные]")
    for k, v in report["input"].items():
        lines.append(f"{k}: {v}")

    lines.append("\n[Изображение]")
    img = report["image_info"]
    lines.append(f"Размер: {img['width']} x {img['height']}")
    lines.append(f"Примерная доля объекта по alpha: {img['foreground_ratio_by_alpha_if_available']:.4f}")
    lines.append(f"Примерная доля не-белого фона: {img['foreground_ratio_by_non_white_heuristic']:.4f}")
    lines.append(f"Комментарий: {img['note']}")

    lines.append("\n[Качество mesh: сгенерированная модель]")
    for k, v in report["mesh_quality"]["pred"].items():
        lines.append(f"{k}: {v}")

    lines.append("\n[Качество mesh: исходная модель]")
    for k, v in report["mesh_quality"]["gt"].items():
        lines.append(f"{k}: {v}")

    lines.append("\n[Сравнение mesh: raw]")
    for k, v in report["mesh_comparison"]["raw"].items():
        lines.append(f"{k}: {v}")

    lines.append("\n[Сравнение mesh: normalized]")
    for k, v in report["mesh_comparison"]["normalized"].items():
        lines.append(f"{k}: {v}")

    lines.append("\n[Геометрические метрики]")
    gm = report["geometry_metrics"]
    important_keys = [
        "chamfer_l1_mean_sum",
        "chamfer_l2_mean_squared_sum",
        "hausdorff_approx",
        "pred_to_gt_mean",
        "gt_to_pred_mean",
        "pred_to_gt_median",
        "gt_to_pred_median",
        "pred_to_gt_p95",
        "gt_to_pred_p95",
        "rms_symmetric",
        "normal_consistency_abs",
        "normal_consistency_oriented",
    ]

    for k in important_keys:
        lines.append(f"{k}: {gm[k]}")

    lines.append("\n[F-score по разным порогам]")
    for t, values in gm["f_scores"].items():
        lines.append(
            f"{t}: precision={values['precision']:.6f}, "
            f"recall={values['recall']:.6f}, "
            f"f_score={values['f_score']:.6f}"
        )

    lines.append("\n[Voxel IoU]")
    for k, v in report["voxel_metrics"].items():
        lines.append(f"{k}: {v}")

    lines.append("\n[Интерпретация]")
    for line in report["interpretation"]:
        lines.append(line)

    if report["saved_files"]["distance_histograms"]:
        lines.append("\n[Сохраненные гистограммы расстояний]")
        for k, v in report["saved_files"]["distance_histograms"].items():
            lines.append(f"{k}: {v}")

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


def parse_thresholds(text: str):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pred", required=True, help="Путь к сгенерированной модели: obj/ply/glb/stl")
    parser.add_argument("--gt", required=True, help="Путь к исходной/эталонной модели")
    parser.add_argument("--image", required=True, help="Путь к входному изображению")
    parser.add_argument("--out_dir", default="eval_one", help="Папка для отчета")
    parser.add_argument("--samples", type=int, default=50000, help="Число точек для сэмплирования поверхности")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--normalization",
        choices=["independent", "gt", "none"],
        default="independent",
        help=(
            "independent — сравнивает форму без учета абсолютного масштаба; "
            "gt — сохраняет ошибку масштаба, если координаты согласованы; "
            "none — без нормализации."
        )
    )
    parser.add_argument("--voxel_resolution", type=int, default=64)
    parser.add_argument(
        "--thresholds",
        default="0.005,0.01,0.02,0.05,0.1",
        help="Пороги для F-score в нормализованных единицах"
    )

    args = parser.parse_args()

    evaluate_one(
        pred_path=args.pred,
        gt_path=args.gt,
        image_path=args.image,
        out_dir=args.out_dir,
        samples=args.samples,
        seed=args.seed,
        normalization=args.normalization,
        voxel_resolution=args.voxel_resolution,
        thresholds=parse_thresholds(args.thresholds)
    )


if __name__ == "__main__":
    main()