"""MCF solver implemented using whirlwind's primal-dual algorithm."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numpy.typing import ArrayLike
from whirlwind.graph import CSRGraph, EdgeList
from whirlwind.network import Network, primal_dual

from ..graph import PlanarGraphInterface, order_points
from ..utils import get_cpu_count, logger
from ._interface import MCFSolverInterface
from .utils import flood_fill, phase_diff, sign_nonzero

__all__ = ["WhirlwindMCFSolver"]


class WhirlwindMCFSolver(MCFSolverInterface):
    """Minimum cost flow solver using whirlwind's primal-dual algorithm.

    Drop-in replacement for ORMCFSolver. Uses a CSRGraph-based Network
    with unit capacity and the primal-dual MCF algorithm from whirlwind.

    Whirlwind releases the GIL during solve, so `residues_to_flows_many`
    can use threads instead of multiprocessing.
    """

    def __init__(self, graph: PlanarGraphInterface):
        """Initialize solver using a planar graph.

        Builds the dual graph and a CSRGraph for use with whirlwind's
        MCF solver.
        """
        self._graph: PlanarGraphInterface = graph
        self._dual_edges: np.ndarray = np.zeros((self.nedges, 2), dtype=np.int32)
        self._dual_edge_dir: np.ndarray = np.zeros((self.nedges, 2), dtype=np.int8)
        self._prepare_dual()
        self._build_csr()
        self._thread_local = threading.local()

    def __getstate__(self) -> dict:
        """Exclude unpicklable C++ objects from pickle state."""
        state = self.__dict__.copy()
        del state["_csr_graph"]
        del state["_thread_local"]
        return state

    def __setstate__(self, state: dict) -> None:
        """Reconstruct C++ objects after unpickling."""
        self.__dict__.update(state)
        self._build_csr()
        self._thread_local = threading.local()

    # ------------------------------------------------------------------ #
    # Properties (same as ORMCFSolver)
    # ------------------------------------------------------------------ #

    @property
    def npoints(self) -> int:
        return self._graph.npoints

    @property
    def points(self) -> np.ndarray:
        return self._graph.points

    @property
    def nedges(self) -> int:
        return len(self._graph.links)

    @property
    def ncycles(self) -> int:
        return len(self._graph.cycles)

    @property
    def edges(self) -> np.ndarray:
        return self._graph.links

    @property
    def cycles(self) -> np.ndarray:
        return self._graph.cycles

    @property
    def cycle_length(self) -> int:
        return len(self.cycles[0])

    @property
    def dual_edges(self) -> np.ndarray:
        return self._dual_edges

    @property
    def dual_edge_dir(self) -> np.ndarray:
        return self._dual_edge_dir

    # ------------------------------------------------------------------ #
    # Dual graph construction (identical to ORMCFSolver)
    # ------------------------------------------------------------------ #

    def _prepare_dual(self) -> None:
        """Identify edges of the dual graph."""
        edge_to_cycles: dict = {tuple(link): [] for link in self.edges}
        for icyc, cycle in enumerate(self.cycles):
            cycsize = len(cycle)
            for ii in range(cycsize):
                jj = (ii + 1) % cycsize
                edge = order_points((cycle[ii], cycle[jj]))
                edge_to_cycles[edge].append(
                    (icyc + 1, sign_nonzero(cycle[jj] - cycle[ii]))
                )

        for ii, kk in enumerate(edge_to_cycles):
            vv = edge_to_cycles[kk]
            ncyc = len(vv)
            if ncyc == 2:
                self._dual_edges[ii, :] = [vv[0][0], vv[1][0]]
                self._dual_edge_dir[ii, :] = [vv[0][1], vv[1][1]]
            elif ncyc == 1:
                self._dual_edges[ii, 0] = vv[0][0]
                self._dual_edge_dir[ii, 0] = vv[0][1]
            else:
                errmsg = (
                    "Planar graph contains edges that are part of more than 2 cycles"
                )
                raise ValueError(errmsg)

    # ------------------------------------------------------------------ #
    # CSRGraph construction
    # ------------------------------------------------------------------ #

    def _build_csr(self) -> None:
        """Build CSRGraph with two directed edges per primal edge.

        For each primal edge i connecting cycles cyc0 and cyc1, creates:
        - Forward edge: cyc0 -> cyc1
        - Backward edge: cyc1 -> cyc0

        Also builds mapping arrays between CSR edge indices and SPURT
        primal edge indices.
        """
        nedges = self.nedges
        cyc0 = self._dual_edges[:, 0]
        cyc1 = self._dual_edges[:, 1]

        # Build edge list with forward + backward edges
        edge_list = EdgeList()
        for i in range(nedges):
            edge_list.add_edge(int(cyc0[i]), int(cyc1[i]))
            edge_list.add_edge(int(cyc1[i]), int(cyc0[i]))

        self._csr_graph = CSRGraph(edge_list)

        # CSRGraph sorts edges by (tail, head). Sort our metadata the same
        # way so that index j in the sorted list corresponds to CSR edge j.
        meta = []
        for i in range(nedges):
            meta.append((int(cyc0[i]), int(cyc1[i]), i, True))
            meta.append((int(cyc1[i]), int(cyc0[i]), i, False))
        meta.sort(key=lambda x: (x[0], x[1]))

        self._csr_to_spurt = np.array([m[2] for m in meta], dtype=np.int32)
        self._csr_is_forward = np.array([m[3] for m in meta])

    # ------------------------------------------------------------------ #
    # Residue computation (identical to ORMCFSolver)
    # ------------------------------------------------------------------ #

    def compute_residues(self, wrapdata: ArrayLike) -> ArrayLike:
        """Compute phase residues for one set of input wrapped data."""
        assert wrapdata.size == self.npoints
        residues = np.zeros(self.ncycles + 1)
        ndim = self.cycle_length
        for col in range(ndim):
            nn = (col + 1) % ndim
            residues[1:] += phase_diff(
                wrapdata[self.cycles[:, col]], wrapdata[self.cycles[:, nn]]
            )
        residues = np.rint(residues / (2 * np.pi)).astype(int)
        residues[0] = -np.sum(residues[1:])
        return residues

    def compute_residues_from_gradients(self, graddata: ArrayLike) -> ArrayLike:
        """Compute residues from edge gradients."""
        assert graddata.size == self.nedges
        cyc0 = self.dual_edges[:, 0]
        cyc1 = self.dual_edges[:, 1]
        cyc0_dir = self.dual_edge_dir[:, 0]
        cyc1_dir = self.dual_edge_dir[:, 1]
        grad_sum = np.zeros(self.ncycles + 1, dtype=np.float32)
        np.add.at(grad_sum, cyc0, cyc0_dir * graddata)
        np.add.at(grad_sum, cyc1, cyc1_dir * graddata)
        residues = np.rint(grad_sum / (2 * np.pi)).astype(int)
        residues[0] = -np.sum(residues[1:])
        return residues

    # ------------------------------------------------------------------ #
    # Solving
    # ------------------------------------------------------------------ #

    def unwrap_one(
        self,
        wrapdata: ArrayLike,
        cost: ArrayLike,
        revcost: ArrayLike | None = None,
    ) -> tuple[ArrayLike, ArrayLike]:
        """Unwrap one set of wrapped phase data."""
        if revcost is None:
            revcost = cost
        residues = self.compute_residues(wrapdata)
        flows = self.residues_to_flows(residues, cost, revcost=revcost)
        unw = flood_fill(wrapdata, self.edges, flows, mode="points")
        return unw, flows

    def residues_to_flows(
        self,
        residues: np.ndarray,
        cost: np.ndarray,
        revcost: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return flows on edges corresponding to the given residues."""
        if not np.any(residues != 0):
            return np.zeros(self.nedges, dtype=np.int32)

        if revcost is None:
            revcost = cost

        csr_cost = self._compute_csr_costs(cost, revcost)
        surplus = residues.astype(np.int32)

        network = self._get_or_create_network(surplus, csr_cost)
        primal_dual(network, maxiter=8)

        return self._extract_flows(network)

    def residues_to_flows_many(
        self,
        residues: np.ndarray,
        cost: np.ndarray,
        revcost: np.ndarray | None = None,
        worker_count: int | None = None,
        chunksize: int | None = 1,
    ) -> np.ndarray:
        """Parallel version of residues_to_flows using threads.

        Whirlwind releases the GIL during solve, so threads give true
        parallelism without the serialization overhead of multiprocessing.
        """
        if (worker_count is None) or (worker_count <= 0):
            worker_count = max(1, get_cpu_count() - 1)

        if revcost is None:
            revcost = cost

        nruns, nresidues = residues.shape
        assert nresidues == self.ncycles + 1

        flows = np.zeros((nruns, self.nedges), dtype=np.int32)

        # Pre-compute CSR costs once (same for all runs)
        csr_cost = self._compute_csr_costs(cost, revcost)

        def _solve_one(ii: int) -> tuple[int, np.ndarray]:
            res = residues[ii]
            if not np.any(res != 0):
                return ii, np.zeros(self.nedges, dtype=np.int32)
            surplus = res.astype(np.int32)
            network = self._get_or_create_network(surplus, csr_cost)
            primal_dual(network, maxiter=8)
            return ii, self._extract_flows(network)

        if worker_count == 1:
            for ii in range(nruns):
                idx, f = _solve_one(ii)
                flows[idx, :] = f
        else:
            logger.info(f"Processing batch of {nruns} with {worker_count} threads")
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                for idx, f in executor.map(_solve_one, range(nruns)):
                    flows[idx, :] = f

        return flows

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _get_or_create_network(
        self, surplus: np.ndarray, csr_cost: np.ndarray
    ) -> Network:
        """Get a thread-local cached Network, or create one.

        On first call per thread, constructs a full Network (building the
        residual graph). On subsequent calls, reuses the existing residual
        graph and only resets surplus, costs, and flow state.
        """
        tls = self._thread_local
        network = getattr(tls, "network", None)
        if network is None:
            tls.network = Network(self._csr_graph, surplus, csr_cost, capacity=1)
            return tls.network
        network.reset(surplus, csr_cost)
        return network

    def _compute_csr_costs(self, cost: np.ndarray, revcost: np.ndarray) -> np.ndarray:
        """Map SPURT per-edge costs to CSR edge-ordered costs.

        Mirrors the cost assignment logic in ORMCFSolver's solve_mcf:
        forward arc cost depends on the edge direction within its first cycle.
        """
        first_dir = self._dual_edge_dir[:, 0]
        fwd_arc_cost = cost * (first_dir == -1) + revcost * (first_dir == 1)
        rev_arc_cost = cost * (first_dir == 1) + revcost * (first_dir == -1)

        spurt_idx = self._csr_to_spurt
        return np.where(
            self._csr_is_forward,
            fwd_arc_cost[spurt_idx],
            rev_arc_cost[spurt_idx],
        ).astype(np.int32)

    def _extract_flows(self, network: Network) -> np.ndarray:
        """Extract SPURT-convention flows from a solved whirlwind Network.

        SPURT flow convention:
            flows[i] = dir[i] * (backward_flow[i] - forward_flow[i])
        where dir is the orientation of the edge in its first cycle.
        """
        nedges = self.nedges

        # Bulk extraction: one C++ call returns flows for all CSR edges
        csr_flows = network.edge_flows()

        # Split into forward/backward flows per SPURT primal edge
        fwd_flow = np.zeros(nedges, dtype=np.int32)
        bwd_flow = np.zeros(nedges, dtype=np.int32)

        fwd_mask = self._csr_is_forward
        spurt_idx = self._csr_to_spurt

        fwd_flow[spurt_idx[fwd_mask]] = csr_flows[fwd_mask]
        bwd_flow[spurt_idx[~fwd_mask]] = csr_flows[~fwd_mask]

        first_dir = self._dual_edge_dir[:, 0].astype(np.int32)
        return first_dir * (bwd_flow - fwd_flow)
