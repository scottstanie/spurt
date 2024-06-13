import logging

import numpy as np
from numpy.typing import ArrayLike

from spurt.workflows.emcf._settings import SolverSettings
from spurt.workflows.emcf._solver import EMCFSolver

logger = logging.getLogger(__name__)

DEFAULT_SOLVER_SETTINGS = SolverSettings(worker_count=4, links_per_batch=100_000)


def unwrap_block(
    wrapped_data_stack: ArrayLike,
    good_pixel_mask: ArrayLike,
    solv_settings: SolverSettings = DEFAULT_SOLVER_SETTINGS,
    flow_mult=1,
) -> np.ndarray:
    from spurt import graph, io, mcf

    good_pixel_xy = np.column_stack(np.where(good_pixel_mask))
    wrapped_input = io.Irreg3DInput(
        wrapped_data_stack[:, good_pixel_mask], xy=good_pixel_xy
    )
    num_time = wrapped_input.shape[0]
    graph_time = graph.Hop3Graph(num_time)
    solver_time = mcf.ORMCFSolver(graph_time)
    # coh = stack.read_temporal_coherence(tile.space)

    # Create spatial graph and solver
    # np.column_stack(np.nonzero(coh > stack.temp_coh_threshold))
    graph_space = graph.DelaunayGraph(good_pixel_xy)
    solver_space = mcf.ORMCFSolver(graph_space)  # type: ignore[abstract]

    # EMCF solver
    solver = EMCFSolver(solver_space, solver_time, solv_settings)
    # wrapped_input = stack.read_tile(tile.space)
    assert wrapped_input.shape[1] == graph_space.npoints
    logger.info(f"Time steps: {solver.nifgs}")
    logger.info(f"Number of points: {solver.npoints}")

    uw_data = solver.unwrap_cube(wrapped_input, flow_mult=flow_mult)
    logger.info("Completed tile: unwrap_cube")

    # Unwrapped data above is always referenced to first pixel
    # since we unwrap gradients. Phase offsets for the first
    # pixel are computed and provided separately. When mosaicking,
    # these offsets need to be added to unwrapped tiles to guarantee
    # integer cycle shifts between tiles.
    ifg_indexes = graph_time.links
    phase_offset = mcf.utils.phase_diff(
        wrapped_input.data[ifg_indexes[:, 0], 0],
        wrapped_input.data[ifg_indexes[:, 1], 0],
    )
    uw_cube = np.full(
        (uw_data.shape[0], *wrapped_data_stack.shape[-2:]), np.nan, dtype=np.float32
    )
    uw_cube[:, good_pixel_xy[:, 0], good_pixel_xy[:, 1]] = uw_data
    return uw_cube, uw_data, phase_offset, graph_space, graph_time, solver
