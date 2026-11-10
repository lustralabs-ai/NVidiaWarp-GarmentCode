# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``wp.mesh_query_edge`` (GarmentCode extension).

The kernel results are compared against a brute-force NumPy reference that runs the
same Woop ray/triangle test in float64 over every triangle, with the same exclusion
rule (triangles containing either end point of the edge are ignored) and the same
acceptance rule (``0 <= t < |x(v2) - x(v1)|``, closest hit wins).
"""

import itertools
import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

# queries whose outcome is decided by less than this margin (in barycentric or
# distance units, geometry is O(1)) are numerically ambiguous between float32 on
# the device and the float64 reference and are not compared
AMBIGUITY_TOL = 1.0e-4


@wp.kernel
def edge_query_kernel(
    mesh: wp.uint64,
    edges: wp.array(dtype=wp.vec2i),
    hit: wp.array(dtype=int),
    t_out: wp.array(dtype=float),
    u_out: wp.array(dtype=float),
    v_out: wp.array(dtype=float),
    sign_out: wp.array(dtype=float),
    normal_out: wp.array(dtype=wp.vec3),
    face_out: wp.array(dtype=int),
):
    tid = wp.tid()
    e = edges[tid]

    # same call pattern as the GarmentCode simulator (out-parameter overload)
    t = float(0.0)
    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)
    normal = wp.vec3()

    if wp.mesh_query_edge(mesh, e[0], e[1], t, face_u, face_v, sign, normal, face_index):
        hit[tid] = 1
        t_out[tid] = t
        u_out[tid] = face_u
        v_out[tid] = face_v
        sign_out[tid] = sign
        normal_out[tid] = normal
        face_out[tid] = face_index
    else:
        hit[tid] = 0
        face_out[tid] = -1


@wp.kernel
def edge_query_struct_kernel(
    mesh: wp.uint64,
    edges: wp.array(dtype=wp.vec2i),
    hit: wp.array(dtype=int),
    t_out: wp.array(dtype=float),
    u_out: wp.array(dtype=float),
    v_out: wp.array(dtype=float),
    sign_out: wp.array(dtype=float),
    normal_out: wp.array(dtype=wp.vec3),
    face_out: wp.array(dtype=int),
):
    tid = wp.tid()
    e = edges[tid]

    q = wp.mesh_query_edge(mesh, e[0], e[1])
    if q.result:
        hit[tid] = 1
        t_out[tid] = q.t
        u_out[tid] = q.u
        v_out[tid] = q.v
        sign_out[tid] = q.sign
        normal_out[tid] = q.normal
        face_out[tid] = q.face
    else:
        hit[tid] = 0
        face_out[tid] = -1


@wp.kernel
def edge_as_ray_kernel(
    mesh: wp.uint64,
    edges: wp.array(dtype=wp.vec2i),
    points: wp.array(dtype=wp.vec3),
    hit: wp.array(dtype=int),
    t_out: wp.array(dtype=float),
):
    # the same rays through mesh_query_ray, i.e. without vertex exclusion
    tid = wp.tid()
    e = edges[tid]
    x1 = points[e[0]]
    d = points[e[1]] - x1
    q = wp.mesh_query_ray(mesh, x1, wp.normalize(d), wp.length(d))
    if q.result:
        hit[tid] = 1
        t_out[tid] = q.t
    else:
        hit[tid] = 0


def run_edge_queries(mesh, edges, device, kernel=edge_query_kernel):
    n = len(edges)
    edges_wp = wp.array(edges.astype(np.int32), dtype=wp.vec2i, device=device)
    hit = wp.zeros(n, dtype=int, device=device)
    t = wp.zeros(n, dtype=float, device=device)
    u = wp.zeros(n, dtype=float, device=device)
    v = wp.zeros(n, dtype=float, device=device)
    sign = wp.zeros(n, dtype=float, device=device)
    normal = wp.zeros(n, dtype=wp.vec3, device=device)
    face = wp.zeros(n, dtype=int, device=device)
    wp.launch(kernel, dim=n, inputs=[mesh.id, edges_wp], outputs=[hit, t, u, v, sign, normal, face], device=device)
    return {
        "hit": hit.numpy().astype(bool),
        "t": t.numpy(),
        "u": u.numpy(),
        "v": v.numpy(),
        "sign": sign.numpy(),
        "normal": normal.numpy(),
        "face": face.numpy(),
    }


# ---------------------------------------------------------------------------
# geometry


def make_grid(n, size, rng, jitter):
    """Jittered n x n grid of quads (2 triangles each) in the xy plane, centered at the origin."""
    xs = np.linspace(-0.5 * size, 0.5 * size, n + 1)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)
    pts += rng.uniform(-jitter, jitter, size=pts.shape) * (size / n)
    tris = []
    for i in range(n):
        for j in range(n):
            a = i * (n + 1) + j
            b = a + 1
            c = a + (n + 1)
            d = c + 1
            tris.append((a, c, b))
            tris.append((b, c, d))
    return pts, np.array(tris, dtype=np.int64)


def make_sphere(n_lat, n_lon, radius, rng, jitter):
    pts = [(0.0, 0.0, radius)]
    for i in range(1, n_lat):
        th = np.pi * i / n_lat
        for j in range(n_lon):
            ph = 2.0 * np.pi * j / n_lon
            pts.append((radius * np.sin(th) * np.cos(ph), radius * np.sin(th) * np.sin(ph), radius * np.cos(th)))
    pts.append((0.0, 0.0, -radius))
    pts = np.array(pts)
    pts *= 1.0 + rng.uniform(-jitter, jitter, size=(len(pts), 1))
    south = len(pts) - 1

    def ring(i, j):
        return 1 + (i - 1) * n_lon + (j % n_lon)

    tris = []
    for j in range(n_lon):
        tris.append((0, ring(1, j), ring(1, j + 1)))
    for i in range(1, n_lat - 1):
        for j in range(n_lon):
            a, b = ring(i, j), ring(i, j + 1)
            c, d = ring(i + 1, j), ring(i + 1, j + 1)
            tris.append((a, c, b))
            tris.append((b, c, d))
    for j in range(n_lon):
        tris.append((ring(n_lat - 1, j), south, ring(n_lat - 1, j + 1)))
    return pts, np.array(tris, dtype=np.int64)


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def make_scene(rng):
    """Three interpenetrating surfaces: a jittered grid, a tilted jittered grid and a jittered sphere."""
    parts = []
    p, t = make_grid(20, 2.0, rng, 0.15)
    parts.append((p, t))
    p, t = make_grid(16, 1.6, rng, 0.15)
    p = p @ rotation((1.0, 0.3, 0.1), 1.1).T + np.array([0.1, -0.05, 0.02])
    parts.append((p, t))
    p, t = make_sphere(14, 24, 0.55, rng, 0.03)
    parts.append((p + np.array([-0.15, 0.2, 0.05]), t))

    points, tris, offset = [], [], 0
    for p, t in parts:
        points.append(p)
        tris.append(t + offset)
        offset += len(p)
    return np.concatenate(points), np.concatenate(tris)


def unique_edges(tris):
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e = np.sort(e, axis=1)
    return np.unique(e, axis=0)


# ---------------------------------------------------------------------------
# brute-force reference


def woop_all(start, dirs, a, b, c):
    """Vectorized float64 port of ``intersect_ray_tri_woop`` (warp/native/intersect.h).

    ``start``/``dirs`` have shape (Q, 3); ``a``/``b``/``c`` have shape (T, 3).
    Returns arrays of shape (Q, T): accepted (as in the native code), t, u, v, det,
    plus ``margin`` = the smallest normalized barycentric (how far inside the triangle).
    """
    q = len(start)
    kz = np.argmax(np.abs(dirs), axis=1)
    kx = (kz + 1) % 3
    ky = (kx + 1) % 3
    swap = dirs[np.arange(q), kz] < 0.0
    kx, ky = np.where(swap, ky, kx), np.where(swap, kx, ky)

    rows = np.arange(q)
    dz = dirs[rows, kz]
    sx = dirs[rows, kx] / dz
    sy = dirs[rows, ky] / dz
    sz = 1.0 / dz

    def project(p):
        rel = p[None, :, :] - start[:, None, :]  # (Q, T, 3)
        px = np.take_along_axis(rel, kx[:, None, None], axis=2)[..., 0]
        py = np.take_along_axis(rel, ky[:, None, None], axis=2)[..., 0]
        pz = np.take_along_axis(rel, kz[:, None, None], axis=2)[..., 0]
        return px - sx[:, None] * pz, py - sy[:, None] * pz, sz[:, None] * pz

    ax, ay, az = project(a)
    bx, by, bz = project(b)
    cx, cy, cz = project(c)

    uu = cx * by - cy * bx
    vv = ax * cy - ay * cx
    ww = bx * ay - by * ax
    det = uu + vv + ww

    inside = ~(((uu < 0) | (vv < 0) | (ww < 0)) & ((uu > 0) | (vv > 0) | (ww > 0)))
    tt = uu * az + vv * bz + ww * cz

    with np.errstate(divide="ignore", invalid="ignore"):
        rcp = 1.0 / det
        u = uu * rcp
        v = vv * rcp
        w = ww * rcp
        t = tt * rcp
    accepted = inside & (det != 0.0) & (t >= 0.0)
    margin = np.minimum(np.minimum(u, v), w)
    return accepted, t, u, v, det, margin


def brute_force(points, tris, edges, exclude=True, chunk=256):
    n = len(edges)
    ref = {
        "hit": np.zeros(n, dtype=bool),
        "t": np.zeros(n),
        "u": np.zeros(n),
        "v": np.zeros(n),
        "sign": np.zeros(n),
        "normal": np.zeros((n, 3)),
        "face": np.full(n, -1),
        "num_hits": np.zeros(n, dtype=int),
        "ambiguous": np.zeros(n, dtype=bool),
    }
    a, b, c = points[tris[:, 0]], points[tris[:, 1]], points[tris[:, 2]]
    normals = np.cross(b - a, c - a)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    for s in range(0, n, chunk):
        e = edges[s : s + chunk]
        x1 = points[e[:, 0]]
        d = points[e[:, 1]] - x1
        max_t = np.linalg.norm(d, axis=1)
        dirs = d / max_t[:, None]

        accepted, t, u, v, det, margin = woop_all(x1, dirs, a, b, c)

        if exclude:
            touches = np.zeros_like(accepted)
            for k in range(3):
                touches |= tris[None, :, k] == e[:, 0:1]
                touches |= tris[None, :, k] == e[:, 1:2]
        else:
            touches = np.zeros_like(accepted)

        valid = accepted & ~touches & (t < max_t[:, None])
        t_valid = np.where(valid, t, np.inf)
        best = np.argmin(t_valid, axis=1)
        rows = np.arange(len(e))
        hit = np.isfinite(t_valid[rows, best])

        # numerically ambiguous: a non-excluded triangle whose plane is crossed inside
        # (or just outside) the segment and whose edge passes within tolerance of the ray,
        # or a hit near t = 0 / t = max_t, or two hits at nearly the same distance
        with np.errstate(invalid="ignore"):
            candidate = ~touches & (det != 0.0)
            # crossing the triangle plane within the segment, close to a triangle edge
            near_segment = candidate & (t > -AMBIGUITY_TOL) & (t < max_t[:, None] + AMBIGUITY_TOL)
            amb = np.any(near_segment & (np.abs(margin) < AMBIGUITY_TOL), axis=1)
            # crossing the triangle close to either end point of the segment
            near_end = (np.abs(t) < AMBIGUITY_TOL) | (np.abs(t - max_t[:, None]) < AMBIGUITY_TOL)
            amb |= np.any(candidate & near_end & (margin > -AMBIGUITY_TOL), axis=1)
            # two hits at nearly the same distance
            t_sorted = np.sort(t_valid, axis=1)
            amb |= hit & (t_sorted[:, 1] - t_sorted[:, 0] < AMBIGUITY_TOL)

        sl = slice(s, s + len(e))
        ref["hit"][sl] = hit
        ref["face"][sl] = np.where(hit, best, -1)
        ref["t"][sl] = np.where(hit, t[rows, best], 0.0)
        ref["u"][sl] = np.where(hit, u[rows, best], 0.0)
        ref["v"][sl] = np.where(hit, v[rows, best], 0.0)
        ref["sign"][sl] = np.where(hit, det[rows, best], 0.0)
        ref["normal"][sl] = np.where(hit[:, None], normals[best], 0.0)
        ref["num_hits"][sl] = np.sum(valid, axis=1)
        ref["ambiguous"][sl] = amb
    return ref


def compare(test, res, ref, points, tris, edges, label):
    ok = ~ref["ambiguous"]
    test.assertGreater(np.count_nonzero(ok), 0.95 * len(edges), f"{label}: too many ambiguous queries")

    mismatch = np.nonzero(ok & (res["hit"] != ref["hit"]))[0]
    test.assertEqual(
        len(mismatch),
        0,
        f"{label}: hit mismatch for edges {edges[mismatch[:10]].tolist()} "
        f"(kernel {res['hit'][mismatch[:10]].tolist()}, reference {ref['hit'][mismatch[:10]].tolist()})",
    )

    h = ok & ref["hit"]
    np.testing.assert_array_equal(res["face"][h], ref["face"][h], err_msg=f"{label}: face")
    np.testing.assert_allclose(res["t"][h], ref["t"][h], atol=1.0e-5, rtol=1.0e-5, err_msg=f"{label}: t")
    np.testing.assert_allclose(res["u"][h], ref["u"][h], atol=1.0e-4, err_msg=f"{label}: bary_u")
    np.testing.assert_allclose(res["v"][h], ref["v"][h], atol=1.0e-4, err_msg=f"{label}: bary_v")
    np.testing.assert_allclose(res["sign"][h], ref["sign"][h], rtol=1.0e-3, atol=1.0e-7, err_msg=f"{label}: sign")
    np.testing.assert_allclose(res["normal"][h], ref["normal"][h], atol=1.0e-5, err_msg=f"{label}: normal")

    # the barycentrics follow the face's vertex order (as used by mesh_eval_position)
    f = tris[res["face"][h]]
    u, v = res["u"][h, None], res["v"][h, None]
    p_bary = points[f[:, 0]] * u + points[f[:, 1]] * v + points[f[:, 2]] * (1.0 - u - v)
    x1 = points[edges[h, 0]]
    d = points[edges[h, 1]] - x1
    p_ray = x1 + d / np.linalg.norm(d, axis=1, keepdims=True) * res["t"][h, None]
    np.testing.assert_allclose(p_bary, p_ray, atol=1.0e-4, err_msg=f"{label}: hit point")

    # misses leave the outputs untouched
    test.assertTrue(np.all(res["face"][~res["hit"]] == -1))

    return np.count_nonzero(h), np.count_nonzero(ok & ~ref["hit"]), np.count_nonzero(h & (ref["num_hits"] >= 2))


def bvh_configs(device):
    constructors = ["sah", "median"] if device.is_cpu else ["sah", "median", "lbvh"]
    if device.supports_cubql:
        constructors.append("cubql")
    return list(itertools.product(constructors, [1, 4]))


# ---------------------------------------------------------------------------
# tests


def test_mesh_query_edge_brute_force(test, device):
    rng = np.random.default_rng(42)
    points, tris = make_scene(rng)
    edges_real = unique_edges(tris)
    edges_rand = rng.integers(0, len(points), size=(3000, 2))
    edges_rand = edges_rand[edges_rand[:, 0] != edges_rand[:, 1]]
    # both orientations of the real edges
    edges = np.concatenate([edges_real, edges_real[:, ::-1], edges_rand])

    ref = brute_force(points, tris, edges)

    for constructor, leaf_size in bvh_configs(device):
        label = f"{device} {constructor} leaf={leaf_size}"
        mesh = wp.Mesh(
            points=wp.array(points, dtype=wp.vec3, device=device),
            indices=wp.array(tris.flatten(), dtype=int, device=device),
            bvh_constructor=constructor,
            bvh_leaf_size=leaf_size,
        )
        for kernel in (edge_query_kernel, edge_query_struct_kernel):
            res = run_edge_queries(mesh, edges, device, kernel)
            num_hit, num_miss, num_multi = compare(test, res, ref, points, tris, edges, f"{label} {kernel.key}")

        # make sure the scene exercises all relevant cases
        test.assertGreater(num_hit, 500)
        test.assertGreater(num_miss, 500)
        test.assertGreater(num_multi, 100)
        n_real = 2 * len(edges_real)
        test.assertGreater(np.count_nonzero(res["hit"][:n_real]), 50)


def test_mesh_query_edge_axis_aligned(test, device):
    # Unjittered, axis-aligned grids: many edges have zero direction components
    # (infinite reciprocal direction in the ray/AABB slab test) and many triangles are
    # exactly coplanar with the query rays.
    rng = np.random.default_rng(3)
    pa, ta = make_grid(12, 1.2, rng, 0.0)  # z = 0 plane
    pb, tb = make_grid(10, 1.0, rng, 0.0)
    pb = pb[:, [0, 2, 1]] + np.array([0.0137, 0.3712, 0.0123])  # vertical plane y = 0.3712
    pc, tc = make_grid(10, 1.0, rng, 0.0)
    pc = pc[:, [2, 0, 1]] + np.array([-0.2213, 0.0091, -0.0217])  # vertical plane x = -0.2213
    points = np.concatenate([pa, pb, pc])
    tris = np.concatenate([ta, tb + len(pa), tc + len(pa) + len(pb)])

    edges_real = unique_edges(tris)
    edges_rand = rng.integers(0, len(points), size=(2000, 2))
    edges_rand = edges_rand[edges_rand[:, 0] != edges_rand[:, 1]]
    edges = np.concatenate([edges_real, edges_real[:, ::-1], edges_rand])

    axis_aligned = np.count_nonzero(np.sum(points[edges[:, 1]] - points[edges[:, 0]] == 0.0, axis=1) >= 2)
    test.assertGreater(axis_aligned, 1000)

    ref = brute_force(points, tris, edges)

    for constructor, leaf_size in bvh_configs(device):
        label = f"{device} {constructor} leaf={leaf_size} axis-aligned"
        mesh = wp.Mesh(
            points=wp.array(points, dtype=wp.vec3, device=device),
            indices=wp.array(tris.flatten(), dtype=int, device=device),
            bvh_constructor=constructor,
            bvh_leaf_size=leaf_size,
        )
        res = run_edge_queries(mesh, edges, device)
        num_hit, num_miss, _ = compare(test, res, ref, points, tris, edges, label)
        test.assertGreater(num_hit, 200)
        test.assertGreater(num_miss, 200)


def test_mesh_query_edge_refit(test, device):
    rng = np.random.default_rng(7)
    points, tris = make_scene(rng)
    edges_real = unique_edges(tris)
    edges_rand = rng.integers(0, len(points), size=(1500, 2))
    edges_rand = edges_rand[edges_rand[:, 0] != edges_rand[:, 1]]
    edges = np.concatenate([edges_real, edges_rand])

    # move the points: rigidly shift the second grid and the sphere, and wiggle everything
    moved = points.copy()
    moved[441:] += np.array([0.12, 0.07, -0.09])
    moved += 0.02 * np.sin(5.0 * points[:, [1, 2, 0]])

    ref_before = brute_force(points, tris, edges)
    ref_after = brute_force(moved, tris, edges)
    changed = (ref_before["hit"] != ref_after["hit"]) | (ref_before["face"] != ref_after["face"])
    test.assertGreater(np.count_nonzero(changed & ~ref_before["ambiguous"] & ~ref_after["ambiguous"]), 50)

    for constructor, leaf_size in bvh_configs(device):
        label = f"{device} {constructor} leaf={leaf_size}"
        points_wp = wp.array(points, dtype=wp.vec3, device=device)
        mesh = wp.Mesh(
            points=points_wp,
            indices=wp.array(tris.flatten(), dtype=int, device=device),
            bvh_constructor=constructor,
            bvh_leaf_size=leaf_size,
        )
        res = run_edge_queries(mesh, edges, device)
        compare(test, res, ref_before, points, tris, edges, f"{label} before refit")

        mesh.points.assign(moved.astype(np.float32))
        mesh.refit()
        res = run_edge_queries(mesh, edges, device)
        compare(test, res, ref_after, moved, tris, edges, f"{label} after refit")


def test_mesh_query_edge_adjacent_excluded(test, device):
    # A fan of triangles around vertex 0 plus a separate triangle (6, 7, 8) that is
    # crossed by the edge 0 -> 5 only when it is long enough.
    points = np.array(
        [
            [0.0, 0.0, 0.0],  # 0: fan center
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.3, 0.2, 1.0],  # 5: off-plane end point
            [-1.0, -1.0, 0.5],  # 6..8: separate triangle in the plane z = 0.5
            [2.0, -1.0, 0.5],
            [-1.0, 2.0, 0.5],
            [0.0, 0.0, 2.0],  # 9: second off-plane end point
        ],
        dtype=np.float64,
    )
    tris = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1], [1, 2, 5], [6, 7, 8]], dtype=np.int64)
    # 0->1: real edge, only touched by adjacent triangles      -> no hit
    # 1->0: same edge reversed                                 -> no hit
    # 0->5: crosses triangle 5 (z = 0.5); triangle 4 contains 5 -> hit face 5
    # 0->9: crosses triangle 5 at t = 0.5                      -> hit face 5
    # 6->7: real edge of the separate triangle                  -> no hit
    edges = np.array([[0, 1], [1, 0], [0, 5], [0, 9], [6, 7]])

    for constructor, leaf_size in bvh_configs(device):
        mesh = wp.Mesh(
            points=wp.array(points, dtype=wp.vec3, device=device),
            indices=wp.array(tris.flatten(), dtype=int, device=device),
            bvh_constructor=constructor,
            bvh_leaf_size=leaf_size,
        )
        res = run_edge_queries(mesh, edges, device)
        label = f"{device} {constructor} leaf={leaf_size}"
        test.assertEqual(res["hit"].tolist(), [False, False, True, True, False], label)
        test.assertEqual(res["face"][2], 5, label)
        test.assertEqual(res["face"][3], 5, label)
        test.assertAlmostEqual(float(res["t"][3]), 0.5, places=5, msg=label)
        np.testing.assert_allclose(res["normal"][3], [0.0, 0.0, 1.0], atol=1.0e-6, err_msg=label)
        test.assertLess(res["sign"][3], 0.0, label)  # ray travels along +z, face normal is +z: back face

        # without the exclusion, rays leaving the fan center hit the adjacent triangles at t = 0
        hit = wp.zeros(len(edges), dtype=int, device=device)
        t_ray = wp.zeros(len(edges), dtype=float, device=device)
        wp.launch(
            edge_as_ray_kernel,
            dim=len(edges),
            inputs=[mesh.id, wp.array(edges.astype(np.int32), dtype=wp.vec2i, device=device), mesh.points, hit, t_ray],
            device=device,
        )
        test.assertEqual(hit.numpy()[2:4].tolist(), [1, 1], label)
        np.testing.assert_allclose(t_ray.numpy()[2:4], 0.0, atol=1.0e-6, err_msg=label)


def test_mesh_query_edge_degenerate(test, device):
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.2, 0.2, -1.0]], dtype=np.float64)
    tris = np.array([[0, 1, 2]], dtype=np.int64)
    mesh = wp.Mesh(
        points=wp.array(points, dtype=wp.vec3, device=device),
        indices=wp.array(tris.flatten(), dtype=int, device=device),
    )
    # v1 == v2 never hits
    res = run_edge_queries(mesh, np.array([[3, 3], [0, 0]]), device)
    test.assertEqual(res["hit"].tolist(), [False, False])


devices = get_test_devices()


class TestMeshQueryEdge(unittest.TestCase):
    pass


add_function_test(
    TestMeshQueryEdge, "test_mesh_query_edge_brute_force", test_mesh_query_edge_brute_force, devices=devices
)
add_function_test(
    TestMeshQueryEdge, "test_mesh_query_edge_axis_aligned", test_mesh_query_edge_axis_aligned, devices=devices
)
add_function_test(TestMeshQueryEdge, "test_mesh_query_edge_refit", test_mesh_query_edge_refit, devices=devices)
add_function_test(
    TestMeshQueryEdge,
    "test_mesh_query_edge_adjacent_excluded",
    test_mesh_query_edge_adjacent_excluded,
    devices=devices,
)
add_function_test(
    TestMeshQueryEdge, "test_mesh_query_edge_degenerate", test_mesh_query_edge_degenerate, devices=devices
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
