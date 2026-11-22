"""Build the brain the look test wraps the weights onto.

    uv run --no-project --with nilearn python web/tools/build_mesh.py

The shape is a real one: FreeSurfer's average cortical surface (fsaverage6,
fetched by nilearn). It is a display surface only; nothing in it comes from
the model. Each hemisphere also has a version inflated to a sphere, which
gives every point an address. The weight picture's squares are laid on the
part of that sphere that faces outward, like lines on a page: one row per
layer from the top of the brain down, six squares along each row. Row and
column sizes are fitted so the squares stay close to square once carried
onto the folded surface (see `fit_grid`). The inner face, hidden between the hemispheres,
gets no weights.

Writes `data/mesh.json` and `data/mesh.bin` (little-endian):
  tiles   float32 [layers * 12, n, n, 7]  position, normal, groove depth
  tissue  float32 [vertices, 7]           the whole cortex, sunk slightly
  faces   uint32  [triangles, 3]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from nilearn import datasets, surface

TILES_PER_HEMISPHERE = 6
SEGMENTS = 56  # quads along a tile edge
FIT_ROUNDS, FIT_STEP, FIT_SEGMENTS = 12, 0.5, 12  # see fit_grid
TILE_INSET = 0.003  # share of a cell left as a seam on each edge
REACH = 2.1  # tiles cover the sphere out to this angle (radians) from the outer side
MM_PER_UNIT = 87.0  # the brain is about two units long
CENTRE_MM = np.array([0.0, -18.0, 15.0])
TILE_LIFT_MM, TISSUE_SINK_MM = 0.25, 0.9


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def vertex_normals(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted outward normals, one per vertex."""
    a, b, c = (points[faces[:, i]] for i in range(3))
    face_normals = np.cross(b - a, c - a)
    normals = np.zeros_like(points)
    for i in range(3):
        np.add.at(normals, faces[:, i], face_normals)
    normals = unit(normals)
    # Triangle winding differs between files; make "out" mean away from the middle.
    if np.mean(np.sum(normals * (points - points.mean(axis=0)), axis=1)) < 0:
        normals = -normals
    return normals


def to_scene(points_mm: np.ndarray) -> np.ndarray:
    """FreeSurfer axes (right, front, up) in mm to scene axes (right, up, front)."""
    return ((points_mm - CENTRE_MM) / MM_PER_UNIT)[:, [0, 2, 1]]


def sphere_frame(sphere: np.ndarray, pial: np.ndarray, side: int) -> np.ndarray:
    """Rows are the sphere directions of front, up and outward for one hemisphere."""

    def toward(axis: int, sign: int) -> np.ndarray:
        order = np.argsort(sign * pial[:, axis])[-300:]
        return unit(sphere[order].mean(axis=0))

    front = unit(toward(1, 1) - toward(1, -1))
    up = toward(2, 1) - toward(2, -1)
    up = unit(up - front * (up @ front))
    outward = np.cross(front, up) * -side
    if outward @ toward(0, side) < 0:
        outward = -outward
    return np.stack([front, up, outward])


class Hemisphere:
    """One side of the cortex, with a lookup from sphere direction to surface."""

    def __init__(self, fsaverage: dict, name: str, side: int) -> None:
        self.side = side
        pial = surface.load_surf_mesh(fsaverage[f"pial_{name}"])
        self.points, self.faces = np.asarray(pial.coordinates, np.float64), np.asarray(pial.faces)
        self.sphere = unit(np.asarray(surface.load_surf_mesh(fsaverage[f"sphere_{name}"]).coordinates))
        self.normals = vertex_normals(self.points, self.faces)
        # How far a point sits inside a fold: overall depth for the broad
        # shadow, curvature for the crisp line at the bottom of each fold.
        depth = np.asarray(surface.load_surf_data(fsaverage[f"sulc_{name}"]), np.float64)
        bend = np.asarray(surface.load_surf_data(fsaverage[f"curv_{name}"]), np.float64)
        deep = np.clip(depth / np.percentile(depth, 98), 0, 1)
        sharp = np.clip(bend / np.percentile(bend, 95), 0, 1)
        self.groove = np.clip(0.55 * deep + 0.7 * sharp, 0, 1)
        self.frame = sphere_frame(self.sphere, self.points, side)

        # For each vertex, the faces that touch it (padded with the first one).
        counts = np.bincount(self.faces.ravel(), minlength=len(self.points))
        self.touching = np.zeros((len(self.points), counts.max()), np.int64)
        fill = np.zeros(len(self.points), np.int64)
        for f, corners in enumerate(self.faces):
            for v in corners:
                self.touching[v, fill[v]] = f
                fill[v] += 1
        for v in range(len(self.points)):
            self.touching[v, fill[v] :] = self.touching[v, 0]

    def at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """[.., 7] position (mm), normal and groove at grid addresses in [-1, 1].

        The grid square is bent into a disc (x toward the front, y up) and the
        disc is wrapped over the outward-facing cap of the sphere.
        """
        front, up, outward = self.frame
        u, v = x * np.sqrt(1 - y * y / 2), y * np.sqrt(1 - x * x / 2)
        radius, around = np.hypot(u, v), np.arctan2(v, u)
        angle = 2 * np.arcsin(np.clip(radius, 0, 1) * np.sin(REACH / 2))  # equal-area
        direction = (
            np.cos(angle)[..., None] * outward
            + (np.sin(angle) * np.cos(around))[..., None] * front
            + (np.sin(angle) * np.sin(around))[..., None] * up
        ).reshape(-1, 3)

        nearest = np.concatenate(
            [np.argmax(chunk @ self.sphere.T, axis=1) for chunk in np.array_split(direction, 64)]
        )
        candidates = self.faces[self.touching[nearest]]  # [q, k, 3]
        corners = self.sphere[candidates]  # [q, k, 3, 3]
        # Barycentric weights of the ray through each candidate triangle.
        weights = np.linalg.solve(
            corners.transpose(0, 1, 3, 2), np.broadcast_to(direction[:, None, :, None], (*corners.shape[:2], 3, 1))
        )[..., 0]
        weights /= weights.sum(axis=-1, keepdims=True)
        best = np.argmax(weights.min(axis=-1), axis=1)
        rows = np.arange(len(direction))
        w = np.clip(weights[rows, best], 0, None)
        w /= w.sum(axis=-1, keepdims=True)
        v = candidates[rows, best]

        position = (self.points[v] * w[..., None]).sum(axis=1)
        normal = unit((self.normals[v] * w[..., None]).sum(axis=1))
        groove = (self.groove[v] * w).sum(axis=1)
        out = np.concatenate([position + normal * TILE_LIFT_MM, normal, groove[:, None]], axis=1)
        return out.reshape(*x.shape, 7)


def tiles(
    hemispheres: dict[int, Hemisphere], layers: int, grid: tuple[np.ndarray, np.ndarray], segments: int
) -> np.ndarray:
    """Every weight tile: tile n shows square n of the picture.

    `grid` holds the edges between columns and between rows, in the grid's
    own -1..1 coordinates: columns run back to front, rows from the top down.
    """
    columns, rows = grid
    along = TILE_INSET + np.linspace(0, 1, segments + 1) * (1 - 2 * TILE_INSET)
    b, a = np.meshgrid(along, along, indexing="ij")
    out = []
    for n in range(layers * TILES_PER_HEMISPHERE * 2):
        layer, slot = divmod(n, TILES_PER_HEMISPHERE * 2)
        side = -1 if slot < TILES_PER_HEMISPHERE else 1
        column = slot % TILES_PER_HEMISPHERE
        x = columns[column] + a * (columns[column + 1] - columns[column])
        y = rows[layer] + b * (rows[layer + 1] - rows[layer])
        out.append(hemispheres[side].at(x, y))
    return np.stack(out)


def stretch(tile_rows: np.ndarray, layers: int) -> np.ndarray:
    """[layers, columns] width over height of each tile on the surface, averaged over hemispheres."""
    points = tile_rows[..., :3]
    width = np.linalg.norm(np.diff(points, axis=2), axis=-1).sum(axis=2).mean(axis=1)
    height = np.linalg.norm(np.diff(points, axis=1), axis=-1).sum(axis=1).mean(axis=1)
    ratio = (width / height).reshape(layers, 2, TILES_PER_HEMISPHERE)
    return np.exp(np.log(ratio).mean(axis=1))


def fit_grid(hemispheres: dict[int, Hemisphere], layers: int) -> tuple[np.ndarray, np.ndarray]:
    """Column and row edges that keep tiles close to square on the folded surface.

    Bending a square grid onto a round cap squeezes its edge rows and columns,
    so equal steps give tiles up to 2.6 times wider than tall. Starting from
    equal steps, each round narrows the columns that came out wide and
    shortens the rows that came out tall, then measures again.
    """
    widths = np.full(TILES_PER_HEMISPHERE, 1.0)
    heights = np.full(layers, 1.0)

    def edges() -> tuple[np.ndarray, np.ndarray]:
        columns = -1 + 2 * np.concatenate([[0], np.cumsum(widths)]) / widths.sum()
        rows = 1 - 2 * np.concatenate([[0], np.cumsum(heights)]) / heights.sum()
        return columns, rows

    for _ in range(FIT_ROUNDS):
        log_ratio = np.log(stretch(tiles(hemispheres, layers, edges(), FIT_SEGMENTS), layers))
        widths *= np.exp(-FIT_STEP * log_ratio.mean(axis=0))
        heights *= np.exp(FIT_STEP * log_ratio.mean(axis=1))
    return edges()


def scene_rows(rows: np.ndarray) -> np.ndarray:
    """Convert [.., 7] rows from mm and FreeSurfer axes to the scene's."""
    flat = rows.reshape(-1, 7)
    scene = np.concatenate([to_scene(flat[:, :3]), flat[:, [3, 5, 4]], flat[:, 6:]], axis=1)
    return scene.reshape(rows.shape).astype("<f4")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent.parent / "public" / "data")
    args = parser.parse_args()

    fsaverage = datasets.fetch_surf_fsaverage("fsaverage6")
    hemispheres = {-1: Hemisphere(fsaverage, "left", -1), 1: Hemisphere(fsaverage, "right", 1)}

    grid = fit_grid(hemispheres, args.layers)
    tile_mm = tiles(hemispheres, args.layers, grid, SEGMENTS)
    ratios = stretch(tile_mm, args.layers)
    print(f"tile width over height: {ratios.min():.2f} to {ratios.max():.2f}, median {np.median(ratios):.2f}")
    tile_rows = scene_rows(tile_mm)
    tissue, faces, offset = [], [], 0
    for hemisphere in hemispheres.values():
        sunk = hemisphere.points - hemisphere.normals * TISSUE_SINK_MM
        tissue.append(np.concatenate([sunk, hemisphere.normals, hemisphere.groove[:, None]], axis=1))
        faces.append(hemisphere.faces + offset)
        offset += len(sunk)
    tissue_rows = scene_rows(np.concatenate(tissue))
    face_rows = np.concatenate(faces).astype("<u4")

    args.out.mkdir(exist_ok=True)
    (args.out / "mesh.bin").write_bytes(tile_rows.tobytes() + tissue_rows.tobytes() + face_rows.tobytes())
    meta = {
        "tiles": len(tile_rows),
        "segments": SEGMENTS,
        "tissue_vertices": len(tissue_rows),
        "triangles": len(face_rows),
        "source": "FreeSurfer fsaverage6 pial surface, via nilearn",
    }
    (args.out / "mesh.json").write_text(json.dumps(meta))
    print(meta)


if __name__ == "__main__":
    main()
