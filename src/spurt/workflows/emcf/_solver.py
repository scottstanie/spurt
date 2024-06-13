"""Implementation of Extended Minimum Cost Flow (EMCF) phase unwrapping."""

from __future__ import annotations

import numpy as np

from spurt.io import Irreg3DInput
from spurt.links import LinkModelInterface
from spurt.mcf import MCFSolverInterface, utils
from spurt.utils import logger

from ._settings import SolverSettings


class EMCFSolver:
    """Implementation of the EMCF algorithm.

    Implements the Extended Minimum Cost Flow (EMCF) algorithm for
    phase unwrapping [1]_. We only implement the solver framework [2]_ as the
    graph generation and cost function implementations are not exactly
    replicable without more details.

    References
    ----------
    .. [1] A. Pepe and R. Lanari, "On the Extension of the Minimum Cost
       Flow Algorithm forPhase Unwrapping of Multitemporal Differential
       SAR Interferograms," in IEEE Transactions on Geoscience and Remote
       Sensing, vol. 44, no. 9, pp. 2374-2383, Sept. 2006,
       doi: 10.1109/TGRS.2006.873207.
    .. [2] K. M. Olsen, M. T. Calef and P. S. Agram, "Contextual uncertainty
           assessments for InSAR-based deformation retrieval using an ensemble
           approach" in Remote Sensing of Environment, 287, p.113456. 2023.
    """

    def __init__(
        self,
        solver_space: MCFSolverInterface,
        solver_time: MCFSolverInterface,
        settings: SolverSettings,
        link_model: LinkModelInterface | None = None,
    ):
        """Spatio-temporal unwrapping.

        Parameters
        ----------
        solver_space: MCFSolverInterface
            MCF Solver interface for spatial graph connecting the stable points.
            Typically a Delaunaytriangulation.
        solver_time: MCFSolverInterface
            MCF Solver interface for temporal graph usually representing
            interferograms in time-Bperp space. Typically a Delaunay triangulation.
        settings: SolverSettings
            Settings to be used for setting up the solver like number of
            workers, link batch size etc.
        model: LinkModel | None
            Per-link model in time used to correct the gradients before
            unwrapping.
        """
        self._solver_space = solver_space
        self._solver_time = solver_time
        self._settings = settings
        self._link_model = link_model

        if link_model is not None:
            errmsg = "Not implemented yet."
            raise NotImplementedError(errmsg)

    @property
    def npoints(self) -> int:
        """Number of points in the spatial network."""
        return self._solver_space.npoints

    @property
    def nlinks(self) -> int:
        """Number of links in the spatial network."""
        return self._solver_space.nedges

    @property
    def nepochs(self) -> int:
        """Number of points in the temporal network."""
        return self._solver_time.npoints

    @property
    def nifgs(self) -> int:
        """Number of links in the temporal network."""
        return self._solver_time.nedges

    @property
    def settings(self) -> SolverSettings:
        """Retrieve settings for the workflow."""
        return self._settings

    @property
    def link_model(self) -> LinkModelInterface | None:
        """Retrieve the link model for the workflow."""
        return self._link_model

    def unwrap_cube(self, wrap_data: Irreg3DInput, flow_mult=1.0) -> np.ndarray:
        """Unwrap a 3D cube of data.

        Parameters
        ----------
        wrap_data: np.ndarray
            2D array of size (nslc, npoints) or (nifg, npoints).

        Returns
        -------
        uw_data: np.ndarray
            2D float32 array of size (nifg, npoints).
        """
        if wrap_data.ndim != 2:
            errmsg = f"Input data is not a 2D array - {wrap_data.ndim}"
            raise ValueError(errmsg)

        if wrap_data.time_dim != 0:
            errmsg = "Time must be first dimension in input stack."
            raise NotImplementedError(errmsg)

        input_is_ifg: bool = False
        if wrap_data.data.shape[0] == self.nepochs:
            input_is_ifg = False
        elif wrap_data.data.shape[0] == self.nifgs:
            input_is_ifg = True
        else:
            errmsg = (
                f"Input size {wrap_data.data.shape[0]} does not match solver"
                f" for {self.nifgs} Ifgs from {self.nepochs} images"
            )
            raise ValueError(errmsg)

        # First unwrap in time to get spatial gradients
        grad_space, flows = self.unwrap_gradients_in_time(
            wrap_data.data, input_is_ifg=input_is_ifg, flow_mult=flow_mult
        )

        # Then unwrap spatial gradients
        return self.unwrap_gradients_in_space(grad_space)

    def unwrap_gradients_in_time(
        self, wrap_data: np.ndarray, *, input_is_ifg: bool, flow_mult: float = 1.0
    ) -> np.ndarray:
        """Temporally unwrap links in parallel.

        The output of this step is the temporally unwrapped phase gradients on
        each link of the spatial graph.
        """
        # First set up temporal cost
        if self.settings.t_cost_type == "constant":
            cost = np.ones(self.nifgs, dtype=int)
        elif self.settings.t_cost_type == "distance":
            cost = utils.distance_costs(
                self._solver_time.points,
                self._solver_time.edges,
                scale=self.settings.t_cost_scale,
            )
        elif self.settings.t_cost_type == "centroid":
            cost = utils.centroid_costs(
                self._solver_time.points,
                self._solver_time.cycles,
                self._solver_time.dual_edges,
                scale=self.settings.t_cost_scale,
            )
        else:
            errmsg = f"Unknown cost type: {self.settings.t_cost_type}"
            raise ValueError(errmsg)

        # Create output array
        grad_space: np.ndarray = np.zeros((self.nifgs, self.nlinks), dtype=np.float32)

        logger.info(f"Temporal: Number of interferograms: {self.nifgs}")
        logger.info(f"Temporal: Number of links: {self.nlinks}")
        logger.info(f"Temporal: Number of cycles: {self._solver_time.ncycles}")

        # Number of batches to process
        nbatches: int = ((self.nlinks - 1) // self.settings.links_per_batch) + 1

        # Iterate over batches
        for bb in range(nbatches):
            i_start = bb * self.settings.links_per_batch
            i_end = min(i_start + self.settings.links_per_batch, self.nlinks)
            links_in_batch = i_end - i_start
            if links_in_batch == 0:
                continue

            # Get indices of points forming links from spatial graph
            inds = self._solver_space.edges[i_start:i_end, :]

            # TODO: Incorporate link_model here when ready
            # Add self._modeled_phase_diff to replace phase_diff

            # Compute spatial gradients for each link
            # If input data is already interferograms
            if input_is_ifg:
                grad_space[:, i_start:i_end] = utils.phase_diff(
                    wrap_data[:, inds[:, 0]], wrap_data[:, inds[:, 1]]
                )
            else:
                logger.info(f"Temporal: Preparing batch {bb + 1}/{nbatches}")
                self._ifg_spatial_gradients_from_slc(
                    wrap_data, inds, grad_space, np.s_[i_start:i_end]
                )

            # Compute residues for each cycle in temporal graph
            # Easier to loop over interferograms here
            ncycles: int = len(self._solver_time.cycles)
            grad_sum: np.ndarray = np.zeros(
                (links_in_batch, ncycles + 1), dtype=np.float32
            )
            for ii in range(self.nifgs):
                # Cycles that ifg contributes to
                cyc = np.abs(self._solver_time.dual_edges[ii])
                cyc_dir = self._solver_time.dual_edge_dir[ii]
                grad_sum[:, cyc[0]] += cyc_dir[0] * grad_space[ii, i_start:i_end]
                if cyc[1] != 0:
                    grad_sum[:, cyc[1]] += cyc_dir[1] * grad_space[ii, i_start:i_end]

            residues = np.rint(grad_sum / (2 * np.pi)).astype(int)
            # Set grounding node
            residues[:, 0] = -np.sum(residues[:, 1:], axis=1)

            # Unwrap the batch
            logger.info(f"Temporal: Unwrapping batch {bb + 1}/{nbatches}")
            flows = self._solver_time.residues_to_flows_many(
                # residues, cost, worker_count=self.settings.worker_count
                residues,
                cost,
                worker_count=1,
            )

            # Update the spatial gradients with estimated flows
            grad_space[:, i_start:i_end] += 2 * np.pi * flows.T * flow_mult
            # grad_space[:, i_start:i_end] += 0

        return grad_space, flows.T

    def unwrap_gradients_in_space(self, grad_space: np.ndarray) -> np.ndarray:
        """Spatially unwrap each interferogram sequentially."""
        # First set up spatial cost
        if self.settings.s_cost_type == "constant":
            cost = np.ones(self.nlinks, dtype=int)
        elif self.settings.s_cost_type == "distance":
            cost = utils.distance_costs(
                self._solver_space.points,
                self._solver_space.edges,
                scale=self.settings.s_cost_scale,
            )
        elif self.settings.s_cost_type == "centroid":
            cost = utils.centroid_costs(
                self._solver_space.points,
                self._solver_space.cycles,
                self._solver_space.dual_edges,
                scale=self.settings.s_cost_scale,
            )
        else:
            errmsg = f"Unknown cost type: {self.settings.s_cost_type}"
            raise ValueError(errmsg)

        # Create output array
        uw_data = np.zeros((self.nifgs, self.npoints), dtype=np.float32)

        logger.info(f"Spatial: Number of interferograms: {self.nifgs}")
        logger.info(f"Spatial: Number of links: {self.nlinks}")
        logger.info(f"Spatial: Number of cycles: {self._solver_space.ncycles}")
        from concurrent.futures import (
            ProcessPoolExecutor,
            as_completed,
        )

        def process_ifg(ii: int):
            ifg_grad = grad_space[ii, :]

            # Compute residues
            residues = self._solver_space.compute_residues_from_gradients(ifg_grad)

            # Unwrap the interferogram - sequential
            flows = self._solver_space.residues_to_flows(residues, cost)

            # Flood fill
            logger.info(f"Finished unwrapping {ii + 1} / {self.nifgs}")
            return utils.flood_fill(
                ifg_grad, self._solver_space.edges, flows, mode="gradients"
            )

        # with ThreadPoolExecutor(max_workers=self.settings.worker_count) as executor:
        #     futures = [executor.submit(process_ifg, ii) for ii in range(self.nifgs)]
        # with ThreadPoolExecutor(max_workers=1) as executor:
        with ProcessPoolExecutor(max_workers=6) as executor:
            futures = [
                executor.submit(
                    process_ifg2, grad_space[ii, :], self._solver_space, cost
                )
                for ii in range(self.nifgs)
            ]
        for ii, fut in enumerate(as_completed(futures)):
            uw_data[ii, :] = fut.result()
            logger.info(f"Spatial: unwrapped {ii + 1} / {self.nifgs}")

        # for ii in range(self.nifgs):
        #     logger.info(f"Spatial: Unwrapping {ii + 1} / {self.nifgs}")
        #     # uw_data[ii, :] = process_ifg(ii)

        #     # # Slice per ifg
        #     # ifg_grad = grad_space[ii, :]

        #     # # Compute residues
        #     # residues = self._solver_space.compute_residues_from_gradients(ifg_grad)

        #     # # Unwrap the interferogram - sequential
        #     # flows = self._solver_space.residues_to_flows(residues, cost)

        #     # # Flood fill
        #     # uw_data[ii, :] = utils.flood_fill(
        #     #     ifg_grad, self._solver_space.edges, flows, mode="gradients"
        #     # )

        return uw_data

    def _ifg_spatial_gradients_from_slc(
        self,
        wrap_data: np.ndarray,
        edges: np.ndarray,
        grad_space: np.ndarray,
        link_slice: slice,
    ) -> None:
        """Compute interferometric spatial gradients from slc data.

        Parameters
        ----------
        wrap_data: np.ndarray
            Wrapped slc data 2D array for whole graph of shape (nslc, npts)
        edges: np.ndarray
            2D array corresponding to edges in spatial graph. These are a
            subset of all links in the graph.
        grad_space: np.ndarray
            Spatial gradient array for the whole graph of shape (nifg, nlinks).
            This array gets updated in place.
        link_slice: slice
            Slice corresponding to edges within the array of all links.
        """
        # Interferogram edges
        ifg_inds = self._solver_time.edges

        # Extract SLC data first
        slc_data0 = wrap_data[:, edges[:, 0]]
        slc_data1 = wrap_data[:, edges[:, 1]]

        # Make interferograms for extracted points
        ifg_data0 = utils.phase_diff(
            slc_data0[ifg_inds[:, 0], :], slc_data0[ifg_inds[:, 1], :]
        )
        ifg_data1 = utils.phase_diff(
            slc_data1[ifg_inds[:, 0], :], slc_data1[ifg_inds[:, 1], :]
        )

        # Update gradient in place
        grad_space[:, link_slice] = utils.phase_diff(ifg_data0, ifg_data1)


def process_ifg2(ifg_grad, solver_space, cost):
    # ifg_grad = grad_space[ii, :]

    # Compute residues
    residues = solver_space.compute_residues_from_gradients(ifg_grad)

    # Unwrap the interferogram - sequential
    flows = solver_space.residues_to_flows(residues, cost)

    # Flood fill
    return utils.flood_fill(ifg_grad, solver_space.edges, flows, mode="gradients")
