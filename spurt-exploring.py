import marimo

__generated_with = "0.6.17"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import plotly.express as px
    import numpy as np

    from dolphin.phase_link import simulate, run_phase_linking
    from dolphin.shp import estimate_neighbors
    from dolphin import io, Strides, HalfWindow

    RNG = np.random.default_rng()
    return (
        HalfWindow,
        RNG,
        Strides,
        estimate_neighbors,
        io,
        mo,
        np,
        px,
        run_phase_linking,
        simulate,
    )


@app.cell
def __(mo):
    mo.md(rf"# Create synthetic deformation")
    return


@app.cell
def __(np, simulate):
    num_time, *shape2d = 30, 128, 128
    num_pixels = np.prod(shape2d)

    C, _ = simulate.simulate_coh(num_time)
    defo_stack = simulate.make_defo_stack((num_time, *shape2d), sigma=20, max_amplitude=10)
    defo_phase = np.exp(1j * defo_stack)
    return C, defo_phase, defo_stack, num_pixels, num_time, shape2d


@app.cell
def __(defo_phase, np, px):
    cmap_wrapped = "twilight"
    px.imshow(np.angle(defo_phase[-1]), color_continuous_scale=cmap_wrapped)
    return cmap_wrapped,


@app.cell
def __(px, shape2d):
    from troposim import turbulence
    spatially_correlated_noise = (turbulence.simulate(shape=shape2d, max_amp=3.5)**2)**(1/3)
    # np.clip(n, 0, 1)
    px.imshow(spatially_correlated_noise)
    return spatially_correlated_noise, turbulence


@app.cell
def __(RNG, np, num_time, shape2d, simulate, spatially_correlated_noise):
    shape3d = (num_time, *shape2d)
    gamma0 = 0.999

    # gamma0_arr = gamma0 * np.ones(shape2d)
    # Make a stip that's low coherence
    # gamma0_arr[:, 25:40] = gamma0_arr[:, 80:95] = 0.01
    # gamma0_arr = RNG.random(shape2d) ** 2

    # gamma0_arr[zero_coh_mask] = 0.01
    gamma0_arr = np.clip(spatially_correlated_noise, 0, 1)

    # prob_zero_cor = 0.8
    # zero_coh_mask = RNG.binomial(1, prob_zero_cor, size=shape2d).astype(bool)
    # zero_coh_mask = spatially_correlated_noise < 0.5
    # scales = RNG.normal(size=shape3d)**2

    # amplitudes = RNG.rayleigh(size=shape3d, scale=scales)
    amplitudes = RNG.rayleigh(size=shape3d, scale=spatially_correlated_noise)
    # amplitudes = np.ones(shape3d)

    gamma_inf: float = 0.1
    Tau0: float = 72
    acq_interval: int = 12

    time = np.arange(0, acq_interval*num_time, acq_interval)

    C_arr = simulate.simulate_coh_stack(
        time, gamma0_arr, gamma_inf, Tau0=Tau0, 
        signal=np.zeros_like(time),
        # amplitudes=amplitudes
    )

    C_arr.shape
    return (
        C_arr,
        Tau0,
        acq_interval,
        amplitudes,
        gamma0,
        gamma0_arr,
        gamma_inf,
        shape3d,
        time,
    )


@app.cell
def __(C_arr, np, num_time):
    for i, c in enumerate(C_arr.reshape(-1, num_time, num_time)):
        np.linalg.cholesky(c)
        print(i)
    return c, i


@app.cell
def __(C_arr, amplitudes, defo_stack, simulate):
    slc_stack = simulate.make_noisy_samples(C=C_arr, defo_stack=defo_stack)
    slc_stack = slc_stack * amplitudes
    slc_stack.shape
    return slc_stack,


@app.cell
def __(amplitudes, np, px):
    # px.imshow(np.abs(slc_stack).mean(axis=0))
    px.imshow(amplitudes[0], zmax=np.percentile(amplitudes, 95))
    return


@app.cell
def __(mo):
    mo.md(rf"# Run Phase Linking")
    return


@app.cell
def __(
    HalfWindow,
    Strides,
    amplitudes,
    estimate_neighbors,
    num_time,
    run_phase_linking,
    slc_stack,
):
    strides = Strides(1, 1)
    half_window = HalfWindow(5, 5)
    neighbor_arrays = estimate_neighbors(halfwin_rowcol=half_window, strides=None, amp_stack=amplitudes, nslc=num_time, alpha=0.05)
    pl_out = run_phase_linking(slc_stack * amplitudes, use_evd=True, half_window=half_window, strides=strides, neighbor_arrays=neighbor_arrays)
    return half_window, neighbor_arrays, pl_out, strides


@app.cell
def __():
    return


@app.cell
def __(cmap_wrapped, np, pl_out, px):
    px.imshow(np.angle(pl_out.cpx_phase[-1]), color_continuous_scale=cmap_wrapped)
    return


@app.cell
def __(pl_out, px):
    px.imshow(pl_out.shp_counts)
    return


@app.cell
def __(pl_out):
    row, col = 55, 65
    pl_out.shp_counts[row, col]
    return col, row


@app.cell
def __(pl_out, px):
    px.imshow(pl_out.temp_coh, zmax=1, zmin=0.5)
    return


@app.cell
def __(pl_out, px):
    px.imshow(pl_out.temp_coh > 0.7)
    return


@app.cell
def __(np, pl_out):
    good_mask = np.logical_and.reduce([
        pl_out.temp_coh > 0.7,
        pl_out.shp_counts > 20,
    ])
    return good_mask,


@app.cell
def __(good_mask, px):
    px.imshow(good_mask)
    return


@app.cell
def __(good_mask, np, pl_out, px):
    px.imshow(np.where(good_mask, np.angle(pl_out.cpx_phase[-1]), np.nan))
    return


@app.cell
def __(mo):
    mo.md(rf"# 3D Unwrapping")
    return


@app.cell
def __(good_mask, np, pl_out):
    import spurt.unwrap
    wrapped_data_stack = pl_out.cpx_phase
    good_pixel_xy = np.column_stack(np.where(good_mask))
    wrapped_input = spurt.io.Irreg3DInput(
        wrapped_data_stack[:, good_mask], xy=good_pixel_xy
    )
    return good_pixel_xy, spurt, wrapped_data_stack, wrapped_input


@app.cell
def __():
    r1, c1 = 48, 66
    r2, c2 = 49, 66
    return c1, c2, r1, r2


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell
def __(good_mask, pl_out, spurt):
    uw_data, phase_offset, graph_space, graph_time, solver = spurt.unwrap.unwrap_block(
        wrapped_data_stack=pl_out.cpx_phase, 
        good_pixel_mask=good_mask,
    )
    return graph_space, graph_time, phase_offset, solver, uw_data


@app.cell
def __(solver, wrapped_input):
    # cost = np.ones(solver.nifgs, dtype=int)
    # grad_space = np.zeros((solver.nifgs, solver.nlinks), dtype='float32')
    grad_space, flows = solver.unwrap_gradients_in_time(wrapped_input.data, input_is_ifg=False)
    print(flows.shape, grad_space.shape)
    return flows, grad_space


@app.cell
def __(flows, grad_space, np):
    grad_space.sum(), np.nonzero(flows), (flows > 0).sum(), (flows < 0).sum()
    return


@app.cell
def __(flows):
    ifg1_flows = flows[0]
    from collections import Counter
    Counter(ifg1_flows)
    return Counter, ifg1_flows


@app.cell
def __(grad_space, np, solver):
    print(np.abs(solver._solver_space.compute_residues_from_gradients(grad_space[0])).sum())
    print(np.abs(solver._solver_space.compute_residues_from_gradients(grad_space[-1])).sum())
    [np.abs(solver._solver_space.compute_residues_from_gradients(g)).sum() for g in grad_space]
    return


@app.cell
def __(mo):
    mo.md(rf"## Plotting the gradients in space which come from `unwrap_gradients_in_time`")
    return


@app.cell
def __():
    return


@app.cell
def __(coords, solver):
    print(solver.nlinks, solver._solver_space.edges.shape, coords.shape)
    solver._solver_space.edges
    return


@app.cell
def __():
    return


@app.cell
def __(graph_space, np, uw_data, wrapped_data_stack):
    coords = graph_space.points.astype(int)
    uw_data.shape, graph_space.points.shape
    out = np.full(wrapped_data_stack.shape, np.nan)
    return coords, out


@app.cell
def __(solver):
    inds = solver._solver_space.edges[:2]

    inds
    return inds,


@app.cell
def __(phase_offset):
    phase_offset
    # phase_offset = mcf.utils.phase_diff(
    #     wrapped_input.data[:, inds[:, 0]],
    #     wrapped_input.data[:, inds[:, 1]]
    # )
    return


@app.cell
def __(graph_time):
    graph_time
    return


@app.cell
def __(graph_space):
    from scipy import spatial
    # fig, ax = spatial.matplotlib.pyplot.subplots()
    # x, y = graph_space.points.T
    # x, y = tri.points.T
    # ax.plot(x, y, 'o', ms=2)
    # ax.triplot(x, y, grad_space._d.simplices.copy())

    # _adjust_bounds(ax, tri.points)
    spatial.delaunay_plot_2d(graph_space._d)
    return spatial,


@app.cell
def __():
    import matplotlib.pyplot as plt
    plt.triplot()
    return plt,


@app.cell
def __(graph_space):
    graph_space.links.shape
    return


if __name__ == "__main__":
    app.run()
