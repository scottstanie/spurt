"""Tests for WhirlwindMCFSolver, mirroring test_ortools.py."""

import numpy as np

import spurt


def wrap(x):
    return x - 2.0 * np.pi * np.round(x / (2.0 * np.pi))


def gen_data_real(seed=32):
    """Generate a sparse 2D dataset."""
    rng = np.random.RandomState(seed)
    npoints = 1000
    points = rng.randn(2 * npoints).reshape((npoints, 2))
    point_data = 0.5 * rng.randn(npoints)
    c = 8.0 * np.pi / np.ptp(points[:, 0])
    point_data += c * points[:, 0]
    graph = spurt.graph.DelaunayGraph(points)
    return graph, point_data


def test_unwrap_one():
    """Test basic 2D unwrapping with whirlwind solver."""
    graph, point_data = gen_data_real()
    edges = graph.links

    solver = spurt.mcf.WhirlwindMCFSolver(graph)
    cost = spurt.mcf.utils.centroid_costs(
        graph.points, solver.cycles, solver.dual_edges
    )
    uwdata, _ = solver.unwrap_one(point_data, cost)

    grads = point_data[edges[:, 1]] - point_data[edges[:, 0]]
    ugrads = uwdata[edges[:, 1]] - uwdata[edges[:, 0]]
    diff = ugrads - grads

    assert solver.npoints == point_data.shape[0]
    assert np.max(np.abs(diff)) > 1
    assert np.ptp(wrap(diff)) < 1.0e-3


def test_snaphu_data():
    """Unwrap a diagonal phase ramp on a regular grid."""
    y, x = np.ogrid[-3:3:256j, -3:3:256j]
    phase = np.pi * (x + y)
    igram = np.exp(1j * phase)

    graph = spurt.graph.Reg2DGraph(igram.shape)
    solver = spurt.mcf.WhirlwindMCFSolver(graph)
    cost = np.ones(solver.edges.shape[0], dtype=int)
    unw, _ = solver.unwrap_one(igram.flatten(), cost)
    unw = unw.reshape(igram.shape)

    mean_diff = np.mean(unw - phase)
    offset = 2.0 * np.pi * np.round(mean_diff / (2.0 * np.pi))
    np.testing.assert_allclose(unw, phase + offset, atol=1e-3)


def test_snaphu_sparse():
    """Unwrap a diagonal phase ramp on a Delaunay triangulation."""
    y, x = np.ogrid[-3:3:256j, -3:3:256j]
    phase = np.pi * (x + y)
    igram = np.exp(1j * phase)

    graph = spurt.graph.DelaunayGraph(spurt.graph.Reg2DGraph(igram.shape).points)
    solver = spurt.mcf.WhirlwindMCFSolver(graph)
    cost = np.ones(solver.edges.shape[0], dtype=int)
    unw, _ = solver.unwrap_one(igram.flatten(), cost)
    unw = unw.reshape(igram.shape)

    mean_diff = np.mean(unw - phase)
    offset = 2.0 * np.pi * np.round(mean_diff / (2.0 * np.pi))
    np.testing.assert_allclose(unw, phase + offset, atol=1e-3)


def test_matches_ortools():
    """Verify whirlwind produces identical flows to OR-Tools."""
    graph, point_data = gen_data_real()

    or_solver = spurt.mcf.ORMCFSolver(graph)
    ww_solver = spurt.mcf.WhirlwindMCFSolver(graph)

    cost = spurt.mcf.utils.centroid_costs(
        graph.points, or_solver.cycles, or_solver.dual_edges
    )
    residues = or_solver.compute_residues(point_data)

    or_flows = or_solver.residues_to_flows(residues, cost)
    ww_flows = ww_solver.residues_to_flows(residues, cost)

    np.testing.assert_array_equal(or_flows, ww_flows)


def test_unwrap_many():
    """Test parallel solving with threads."""
    graph, point_data = gen_data_real()
    solver = spurt.mcf.WhirlwindMCFSolver(graph)

    cost = spurt.mcf.utils.centroid_costs(
        graph.points, solver.cycles, solver.dual_edges
    )
    resid = solver.compute_residues(point_data)

    ncopies = 20
    residues = np.tile(resid, (ncopies, 1))

    flows = solver.residues_to_flows_many(residues, cost, worker_count=4)
    assert np.ptp(np.ptp(flows, axis=0)) == 0


def test_unwrap_many_oneworker():
    """Test sequential solving."""
    graph, point_data = gen_data_real()
    solver = spurt.mcf.WhirlwindMCFSolver(graph)

    cost = spurt.mcf.utils.centroid_costs(
        graph.points, solver.cycles, solver.dual_edges
    )
    resid = solver.compute_residues(point_data)

    ncopies = 4
    residues = np.tile(resid, (ncopies, 1))

    flows = solver.residues_to_flows_many(residues, cost, worker_count=1)
    assert np.ptp(np.ptp(flows, axis=0)) == 0


def test_residues():
    """Verify residue computation matches ORMCFSolver."""
    graph, point_data = gen_data_real()
    or_solver = spurt.mcf.ORMCFSolver(graph)
    ww_solver = spurt.mcf.WhirlwindMCFSolver(graph)

    grads = spurt.mcf.utils.phase_diff(
        point_data[or_solver.edges[:, 0]], point_data[or_solver.edges[:, 1]]
    )

    or_pts = or_solver.compute_residues(point_data)
    ww_pts = ww_solver.compute_residues(point_data)
    np.testing.assert_array_equal(or_pts, ww_pts)

    or_grads = or_solver.compute_residues_from_gradients(grads)
    ww_grads = ww_solver.compute_residues_from_gradients(grads)
    np.testing.assert_array_equal(or_grads, ww_grads)
