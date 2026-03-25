import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Demo 3: 2D Crystal Buckling — Bifurcation & Branch Switching

        A 2D hexagonal Lennard-Jones crystal is compressed uniaxially.
        At a critical strain (~14.5 %) the **smallest eigenvalue of the
        Hessian crosses zero** — the symmetric (uniformly compressed) solution
        becomes unstable and two equivalent **buckled** configurations emerge.
        This is a **pitchfork bifurcation**.

        The demo shows how to:
        1. Monitor the Hessian eigenvalue during quasi-static loading
        2. Detect the bifurcation point
        3. Continue the **symmetric** (now unstable) branch past the
           bifurcation — the Krylov solver follows unstable saddle-point
           equilibria that LAMMPS minimisation cannot reach
        4. Switch to the **buckled** branch by perturbing along the
           unstable eigenvector and re-minimising
        5. Compare the energy of both branches — the pitchfork diagram
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from lammps import lammps
    from LACT import atom_cont_system
    from scipy.sparse.linalg import LinearOperator, eigsh
    return LinearOperator, atom_cont_system, eigsh, lammps, mo, np, plt


@app.cell
def _(LinearOperator, eigsh, np):
    def get_smallest_eigen_2d(sys_obj, Y):
        """Smallest eigenvalue & eigenvector of the 2D-projected Hessian."""
        sys_obj.pass_ext_variable_info(Y)
        sys_obj.lmp.command("run 0")
        _f0 = (
            np.array(sys_obj.lmp.gather_atoms("f", 1, 3))
            .reshape(sys_obj.natoms, 3)[:, :2].flatten().copy()
        )
        _eps = 1e-5

        def _Hv(v2d):
            v3d = np.zeros(3 * sys_obj.natoms)
            v3d[0::3] = v2d[0::2]
            v3d[1::3] = v2d[1::2]
            _Yp = Y.copy()
            _Yp[:-1] += _eps * v3d
            sys_obj.pass_ext_variable_info(_Yp)
            sys_obj.lmp.command("run 0")
            _f1 = (
                np.array(sys_obj.lmp.gather_atoms("f", 1, 3))
                .reshape(sys_obj.natoms, 3)[:, :2].flatten().copy()
            )
            return -(_f1 - _f0) / _eps

        _ndof = 2 * sys_obj.natoms
        _H = LinearOperator((_ndof, _ndof), matvec=_Hv, dtype=float)
        _vals, _vecs = eigsh(_H, k=3, which="SA", tol=1e-6, maxiter=1000)
        _idx = np.argsort(_vals)
        for _i in _idx:
            if abs(_vals[_i]) > 0.1:
                return _vals[_i], _vecs[:, _i]
        return _vals[_idx[-1]], _vecs[:, _idx[-1]]

    return (get_smallest_eigen_2d,)


@app.cell
def _(atom_cont_system, lammps):
    _NX = 4

    def make_crystal():
        """Create a 2D hex LJ crystal, relaxed to zero stress."""
        _lmp = lammps(cmdargs=["-screen", "none"])
        _lmp.commands_string(
            f"""
            units         lj
            dimension     2
            boundary      p p p
            atom_style    atomic
            atom_modify   map yes
            lattice       hex 0.9165
            region        box block 0 {_NX} 0 {_NX} -0.5 0.5
            create_box    1 box
            create_atoms  1 box
            mass          1 1.0
            pair_style    lj/cut 2.5
            pair_coeff    1 1 1.0 1.0 2.5
            fix boxrelax all box/relax iso 0.0 vmax 0.001
            minimize 0 1e-12 50000 50000
            unfix boxrelax
            compute forces all property/atom fx fy fz
            compute ids all property/atom id
            run 0
            """
        )
        _box = _lmp.extract_box()
        _xlo, _xhi = _box[0][0], _box[1][0]
        _Lx = _xhi - _xlo

        def _update(strain):
            _dx = _Lx * strain / 2
            return f"change_box all x final {_xlo+_dx} {_xhi-_dx} units box"

        return atom_cont_system(_lmp, _update), _lmp

    return (make_crystal,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Symmetric branch — QS + eigenvalue monitoring + continuation

        Compress the crystal in small strain increments. At each step we
        compute the smallest Hessian eigenvalue. When it crosses zero the
        symmetric solution has become unstable. We then extend the symmetric
        branch **past** the bifurcation using arclength continuation — the
        Krylov solver can follow unstable equilibria where LAMMPS minimisation
        would jump to the buckled state.
        """
    )
    return


@app.cell
def _(get_smallest_eigen_2d, make_crystal, np):
    sym_sys, _sym_lmp = make_crystal()

    # QS to just before the bifurcation (~strain 0.135)
    sym_sys.quasi_static_run(0.0, 0.005, 28, verbose=False)
    n_qs_sym = len(sym_sys.data["Y_s"])

    # Eigenvalue at each QS point
    qs_strains = []
    qs_eigvals = []
    for _Y in sym_sys.data["Y_s"]:
        _ev, _ = get_smallest_eigen_2d(sym_sys, _Y)
        qs_strains.append(_Y[-1])
        qs_eigvals.append(_ev)

    qs_strains = np.array(qs_strains)
    qs_eigvals = np.array(qs_eigvals)

    # Continue symmetric branch past the bifurcation
    sym_sys.continuation_run(
        n_iter=50,
        ds_default=0.1,
        ds_smallest=0.001,
        ds_largest=0.5,
        verbose=True,
        checkpoint_freq=0,
        cont_target=0.20,
        target_tol=0.005,
    )
    sym_sys.compute_energies()
    sym_strains = np.array([_Y[-1] for _Y in sym_sys.data["Y_s"]])
    sym_energies = np.array(sym_sys.data["energies"])
    return n_qs_sym, qs_eigvals, qs_strains, sym_energies, sym_strains, sym_sys


@app.cell
def _(n_qs_sym, plt, qs_eigvals, qs_strains):
    fig_eig, ax_eig = plt.subplots(figsize=(7, 4))
    ax_eig.plot(qs_strains, qs_eigvals, "o-", ms=4, lw=1.5)
    ax_eig.axhline(0, color="k", ls="--", lw=0.8)
    ax_eig.axvline(qs_strains[n_qs_sym - 1], color="r", ls=":", lw=1,
                   label=f"Bifurcation near strain = {qs_strains[n_qs_sym - 1]:.3f}")
    ax_eig.set_xlabel("Uniaxial strain")
    ax_eig.set_ylabel("Smallest Hessian eigenvalue")
    ax_eig.set_title("Stability of the symmetric branch under uniaxial compression")
    ax_eig.legend()
    ax_eig.grid(True, alpha=0.3)
    fig_eig.tight_layout()
    fig_eig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Buckled branch — perturb & trace

        Well past the bifurcation (strain ~16 %), the instability is strong
        enough for the buckled state to be clearly separated from the
        symmetric one. We apply a deterministic sinusoidal y-perturbation
        and re-minimise — the system falls into the buckled minimum.
        We then trace the buckled branch with quasi-static steps.
        """
    )
    return


@app.cell
def _(make_crystal, np):
    buck_sys, _buck_lmp = make_crystal()

    # QS well past bifurcation — CG stays on the symmetric saddle branch
    buck_sys.quasi_static_run(0.0, 0.005, 39, verbose=False)
    _n = buck_sys.natoms
    _U0 = buck_sys.U_0

    # Deterministic perturbation: sinusoidal y-displacement.
    # This avoids the random sign issue with eigsh eigenvectors and
    # produces identical results on every run.
    _box = _buck_lmp.extract_box()
    _Lx = _box[1][0] - _box[0][0]
    _v3d = np.zeros(3 * _n)
    for _i in range(_n):
        _v3d[3 * _i + 1] = 0.1 * np.sin(2 * np.pi * _U0[_i, 0] / _Lx)

    # Perturb + minimise at two high strains to seed the continuation
    _buckled_Ys = []
    for _step in [38, 37]:  # strains 0.190, 0.185 (high → low)
        _Y_s = buck_sys.data["Y_s"][_step].copy()
        _Y_p = _Y_s.copy()
        _Y_p[:-1] += _v3d
        buck_sys.pass_ext_variable_info(_Y_p)
        _buck_lmp.command("run 0")
        _buck_lmp.command("min_style cg")
        _buck_lmp.command("minimize 0 1e-10 10000 10000")
        _X_b, _ = buck_sys.get_positions_from_lammps()
        _buckled_Ys.append(
            np.append((_X_b - _U0).flatten(), _Y_s[-1])
        )

    # Seed continuation: higher strain first → tangent toward lower strain.
    _Y_buck = _buckled_Ys[0]
    _Y_buck2 = _buckled_Ys[1]
    # Seed continuation: higher strain first so the tangent points toward
    # lower strain → traces the buckled branch backward toward the
    # bifurcation point.
    buck_sys.data["Y_s"] = [_Y_buck, _Y_buck2]
    buck_sys.data["ds_s"] = []

    buck_sys.continuation_run(
        n_iter=300,
        ds_default=0.002,
        ds_smallest=1e-6,
        ds_largest=0.02,
        verbose=True,
        checkpoint_freq=0,
        cont_target=0.14,
        target_tol=0.01,
    )

    # Drop the first seed point whose energy is unreliable due to
    # the change_box float-precision round-trip.
    buck_sys.data["Y_s"] = buck_sys.data["Y_s"][1:]
    buck_sys.compute_energies()
    buck_strains = np.array([_Y[-1] for _Y in buck_sys.data["Y_s"]])
    buck_energies = np.array(buck_sys.data["energies"])
    return buck_energies, buck_strains, buck_sys


@app.cell
def _(buck_energies, buck_strains, n_qs_sym, plt, sym_energies, sym_strains):
    fig_en, ax_en = plt.subplots(figsize=(7, 5))

    # Symmetric: QS (stable) + continuation (unstable past bifurcation)
    ax_en.plot(sym_strains[:n_qs_sym], sym_energies[:n_qs_sym],
               "s-", ms=5, lw=1.5, color="C1", label="Symmetric (stable, QS)")
    ax_en.plot(sym_strains[n_qs_sym:], sym_energies[n_qs_sym:],
               "s--", ms=4, lw=1.5, color="C1", alpha=0.5,
               label="Symmetric (unstable, continuation)")

    # Buckled branch
    ax_en.plot(buck_strains, buck_energies,
               "o-", ms=5, lw=1.5, color="C0", label="Buckled branch (QS)")

    ax_en.set_xlabel("Uniaxial strain")
    ax_en.set_ylabel("Energy per atom")
    ax_en.set_title("Pitchfork bifurcation: energy of symmetric vs. buckled branch")
    ax_en.legend()
    ax_en.grid(True, alpha=0.3)
    fig_en.tight_layout()
    fig_en
    return


@app.cell
def _(buck_sys, np, plt, sym_sys):
    _n = sym_sys.natoms
    _U0_sym = sym_sys.U_0

    fig_cfg, axes = plt.subplots(1, 3, figsize=(12, 4))

    _snapshots = [
        (sym_sys, 0, "Undeformed"),
        (sym_sys, len(sym_sys.data["Y_s"]) - 1, "Symmetric (compressed)"),
        (buck_sys, len(buck_sys.data["Y_s"]) - 1, "Buckled"),
    ]
    for ax, (_sys, _idx, _title) in zip(axes, _snapshots):
        _Y = _sys.data["Y_s"][_idx]
        _dev = _Y[:-1].reshape(_n, 3)
        _pos = _sys.U_0 + _dev

        ax.scatter(_pos[:, 0], _pos[:, 1], s=80, c="C0", edgecolors="k", lw=0.5)

        # Displacement arrows from the undeformed reference
        if _idx > 0:
            for _j in range(_n):
                _d = _pos[_j, :2] - _U0_sym[_j, :2]
                if np.linalg.norm(_d) > 1e-3:
                    ax.annotate(
                        "", xy=(_pos[_j, 0], _pos[_j, 1]),
                        xytext=(_U0_sym[_j, 0], _U0_sym[_j, 1]),
                        arrowprops=dict(arrowstyle="->", color="r", lw=0.8),
                    )

        ax.set_title(f"{_title}\n(strain = {_Y[-1]:.3f})")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

    fig_cfg.suptitle("Atomic configurations", fontsize=13, y=1.02)
    fig_cfg.tight_layout()
    fig_cfg
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Interpreting the results

        **Eigenvalue plot:** The smallest Hessian eigenvalue decreases under
        uniaxial compression and crosses zero at ~14.5 % strain. Below this
        strain the symmetric solution is the unique energy minimum; above it
        the symmetric state is a saddle point.

        **Energy plot:** The two branches share the same energy up to the
        bifurcation, then diverge. The buckled branch (blue) has **lower
        energy** — it is the new stable equilibrium. The symmetric branch
        (orange dashed) is continued past the bifurcation by LACT's
        arclength solver, which can follow unstable equilibria where
        LAMMPS minimisation would jump to the buckled state.

        **Configuration snapshots:** Red arrows show displacements from the
        undeformed reference. In the symmetric compressed state, atoms move
        only along the compression axis. In the buckled state, atoms also
        shift perpendicular to the compression, breaking the crystal's
        reflection symmetry — the hallmark of a pitchfork bifurcation.
        """
    )
    return


if __name__ == "__main__":
    app.run()
