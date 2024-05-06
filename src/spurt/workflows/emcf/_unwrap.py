import json
from pathlib import Path

import h5py
import numpy as np

import spurt

from ._settings import GeneralSettings, SolverSettings
from ._solver import EMCFSolver

logger = spurt.utils.logger


def unwrap_tiles(
    stack: spurt.io.SLCStackReader,
    gen_settings: GeneralSettings,
    solv_settings: SolverSettings,
) -> None:
    """Unwrap each tile and save to h5."""
    # I/O files
    pdir = Path(gen_settings.output_folder)
    json_name = pdir / "tiles.json"
    tile_file_tmpl = str(pdir / "uw_tile_{}.h5")

    # Temporal graph
    # TODO: Generalize later to a generic graph
    n_sar = len(stack.slc_files)
    g_time = spurt.graph.Hop3Graph(n_sar)
    s_time = spurt.mcf.ORMCFSolver(g_time)  # type: ignore[abstract]

    with json_name.open(mode="r") as fid:
        tiledata = json.load(fid)

    # Iterate over tiles
    for tt in range(len(tiledata["tiles"])):
        tfname = tile_file_tmpl.format(f"{tt + 1:02d}")
        if Path(tfname).is_file():
            logger.info(f"Tile {tt+1} already processed. Skipping...")
            continue

        # Select valid pixels from coherence file
        logger.info(f"Processing tile: {tt+1}")
        space = _bounds_to_space(tiledata["tiles"][tt]["bounds"])
        coh = stack.read_temporal_coherence(space)

        # Create spatial graph and solver
        g_space = spurt.graph.DelaunayGraph(
            np.column_stack(np.nonzero(coh > stack.temp_coh_threshold))
        )
        s_space = spurt.mcf.ORMCFSolver(g_space)  # type: ignore[abstract]

        # EMCF solver
        solver = EMCFSolver(s_space, s_time, settings=solv_settings)
        wrap_data = stack.read_tile(space)
        assert wrap_data.shape[1] == g_space.npoints
        logger.info(f"Time steps: {solver.nifgs}")
        logger.info(f"Number of points: {solver.npoints}")

        uw_data = solver.unwrap_cube(wrap_data.data)
        logger.info(f"Completed tile: {tt+1}")

        _dump_tile_to_h5(tfname, uw_data, g_space, tiledata["tiles"][tt])
        logger.info(f"Wrote tile {tt + 1} to {tfname}")
        wrap_data = None
        uw_data = None


def _dump_tile_to_h5(
    fname: str, uw: np.ndarray, gspace: spurt.graph.PlanarGraphInterface, tile: dict
) -> None:
    with h5py.File(fname, "w") as fid:
        fid["uw_data"] = uw
        fid["points"] = gspace.points
        fid["tile"] = np.array(tile["bounds"]).astype(np.int32)


def _bounds_to_space(bounds):
    """Shapely style bounds to numpy style slices."""
    return (slice(bounds[0], bounds[2]), slice(bounds[1], bounds[3]))
