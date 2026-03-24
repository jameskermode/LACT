import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Demo 2: Morse Chain — Which Bond Snaps?

        A chain of 6 atoms connected by Morse bonds is pulled apart by a
        force applied to the right end. Each nearest-neighbour bond has a
        different well depth $D$, giving each a different maximum sustainable
        tension $F_{\max} = D\alpha/2$. The **weakest bond** (smallest $D$)
        reaches its fold first and snaps.

        Arclength continuation traces the force–extension S-curve through the
        fold, revealing the full unstable branch that load-stepping cannot
        reach. The bond-length plot confirms that only the weakest bond
        snaps — the others remain near equilibrium.
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
    return atom_cont_system, lammps, mo, np, plt


@app.cell
def _(atom_cont_system, lammps):
    # --- Parameters ---
    n_atoms = 6
    alpha = 6.0       # Morse width parameter
    r0 = 1.12         # Morse equilibrium distance
    # Well depths per bond — ordered weakest to strongest
    bond_info = [
        ((1, 2), 0.85),  # weakest — snaps first
        ((2, 3), 0.90),
        ((3, 4), 0.95),
        ((4, 5), 1.00),
        ((5, 6), 1.05),  # strongest
    ]

    # Helper to create a fresh chain system
    def make_chain():
        lmp = lammps(cmdargs=["-screen", "none"])
        lmp.commands_string(
            f"""
            units         lj
            dimension     3
            boundary      f f f
            atom_style    atomic
            atom_modify   map yes

            region box block -50.0 50.0 -50.0 50.0 -50.0 50.0
            create_box {n_atoms} box
            mass * 1.0

            create_atoms 1 single {0 * r0:.6f} 0.0 0.0
            create_atoms 2 single {1 * r0:.6f} 0.0 0.0
            create_atoms 3 single {2 * r0:.6f} 0.0 0.0
            create_atoms 4 single {3 * r0:.6f} 0.0 0.0
            create_atoms 5 single {4 * r0:.6f} 0.0 0.0
            create_atoms 6 single {5 * r0:.6f} 0.0 0.0
            reset_atom_ids sort yes

            pair_style morse 5.0
            pair_coeff * * 0.001 {alpha} {r0} 0.001
            pair_coeff 1 2 {bond_info[0][1]} {alpha} {r0} 5.0
            pair_coeff 2 3 {bond_info[1][1]} {alpha} {r0} 5.0
            pair_coeff 3 4 {bond_info[2][1]} {alpha} {r0} 5.0
            pair_coeff 4 5 {bond_info[3][1]} {alpha} {r0} 5.0
            pair_coeff 5 6 {bond_info[4][1]} {alpha} {r0} 5.0

            group left_end id 1
            group right_end id 6
            fix freeze left_end setforce 0.0 0.0 0.0
            fix pull right_end addforce 0.0 0.0 0.0
            fix_modify pull energy yes

            compute forces all property/atom fx fy fz
            compute ids all property/atom id
            """
        )

        def update_command(force):
            return f"""
            unfix pull
            fix pull right_end addforce {force} 0.0 0.0
            fix_modify pull energy yes
            """

        return atom_cont_system(lmp, update_command)

    return alpha, bond_info, make_chain, n_atoms, r0


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Quasi-static load-stepping (for comparison)

        Ramp the applied force in fine steps from 0 to 3.0. This works up to
        the fold ($F_{\max} = 0.85 \times 6/2 = 2.55$), where the LAMMPS
        minimiser loses the atom — the bond snaps discontinuously and the
        solution is lost.
        """
    )
    return


@app.cell
def _(make_chain, n_atoms, np):
    # Quasi-static only: force 0 → 3.0 in steps of 0.1 (crashes past fold)
    qs_chain = make_chain()
    try:
        qs_chain.quasi_static_run(0.0, 0.1, 30, verbose=False)
    except Exception:
        pass  # LAMMPS crashes past the fold — keep the points we got

    qs_U0 = qs_chain.U_0
    qs_extensions = []
    qs_forces = []
    qs_bonds = []
    for _Y in qs_chain.data["Y_s"]:
        _pos = qs_U0 + _Y[:-1].reshape(n_atoms, 3)
        qs_extensions.append(_pos[-1, 0] - qs_U0[-1, 0])
        qs_forces.append(_Y[-1])
        qs_bonds.append([_pos[i + 1, 0] - _pos[i, 0] for i in range(n_atoms - 1)])
    qs_extensions = np.array(qs_extensions)
    qs_forces = np.array(qs_forces)
    qs_bonds = np.array(qs_bonds)
    return qs_bonds, qs_extensions, qs_forces


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Arclength continuation

        A short quasi-static ramp seeds the path, then `continuation_run`
        traces smoothly through the fold and onto the unstable branch.
        """
    )
    return


@app.cell
def _(make_chain, n_atoms, np):
    cont_chain = make_chain()
    cont_chain.quasi_static_run(0.0, 0.5, 5, verbose=True)
    n_quasi_static = len(cont_chain.data["Y_s"])

    cont_chain.continuation_run(
        n_iter=200,
        ds_default=0.1,
        ds_smallest=0.001,
        ds_largest=0.5,
        verbose=True,
        checkpoint_freq=0,
        cont_target=1.0,
        target_tol=0.1,
    )

    cont_U0 = cont_chain.U_0
    cont_extensions = []
    cont_forces = []
    cont_bonds = []
    for _Y in cont_chain.data["Y_s"]:
        _pos = cont_U0 + _Y[:-1].reshape(n_atoms, 3)
        cont_extensions.append(_pos[-1, 0] - cont_U0[-1, 0])
        cont_forces.append(_Y[-1])
        cont_bonds.append([_pos[i + 1, 0] - _pos[i, 0] for i in range(n_atoms - 1)])
    cont_extensions = np.array(cont_extensions)
    cont_forces = np.array(cont_forces)
    cont_bonds = np.array(cont_bonds)

    # Theoretical fold forces per bond
    return cont_bonds, cont_extensions, cont_forces, n_quasi_static


@app.cell
def _(alpha, bond_info):
    F_max_vals = [(ids, D, D * alpha / 2) for ids, D in bond_info]
    return (F_max_vals,)


@app.cell
def _(cont_extensions, cont_forces, n_quasi_static, plt, qs_extensions, qs_forces):
    fig, ax = plt.subplots(figsize=(7, 5))

    # Quasi-static (load-stepping) — shows failure past the fold
    ax.plot(qs_extensions, qs_forces,
            "s", ms=5, color="C1", alpha=0.7, label="Quasi-static (load-stepping)")

    # Continuation seed + arclength
    ax.plot(cont_extensions[:n_quasi_static], cont_forces[:n_quasi_static],
            "D", ms=5, color="C0", alpha=0.5)
    ax.plot(cont_extensions[n_quasi_static:], cont_forces[n_quasi_static:],
            "o-", ms=3, lw=1.5, color="C0", label="Arclength continuation")

    ax.set_xlabel("Extension (displacement of atom 6)")
    ax.set_ylabel("Applied force")
    ax.set_title("Force–extension through the first fold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig
    return


@app.cell
def _(F_max_vals, alpha, cont_bonds, cont_forces, n_quasi_static, np, plt, qs_bonds, qs_forces, r0):
    fig2, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    _r_fold = r0 + np.log(2) / alpha

    for _idx, ((i, j), D, Fmax) in enumerate(F_max_vals):
        _c = f"C{_idx}"
        # Quasi-static
        ax_top.plot(qs_forces, qs_bonds[:, _idx],
                    "s", ms=4, color=_c, alpha=0.5)
        # Continuation
        ax_top.plot(cont_forces[n_quasi_static:], cont_bonds[n_quasi_static:, _idx],
                    "-", lw=1.5, color=_c,
                    label=f"Bond {i}–{j} ($D$={D:.2f}, $F_{{max}}$={Fmax:.2f})")

    ax_top.axhline(_r_fold, color="k", ls="--", lw=0.8, alpha=0.5,
                   label=f"Fold at $r^*$={_r_fold:.3f}")
    ax_top.set_ylabel(r"Bond length $r$")
    ax_top.set_title("Bond lengths vs. applied force (squares = quasi-static, lines = continuation)")
    ax_top.legend(fontsize=7, loc="upper left")
    ax_top.grid(True, alpha=0.3)

    # Deviation from equilibrium
    for _idx, ((i, j), D, _Fmax) in enumerate(F_max_vals):
        _c = f"C{_idx}"
        ax_bot.plot(qs_forces, qs_bonds[:, _idx] - r0,
                    "s", ms=4, color=_c, alpha=0.5)
        ax_bot.plot(cont_forces[n_quasi_static:], cont_bonds[n_quasi_static:, _idx] - r0,
                    "-", lw=1.5, color=_c, label=f"Bond {i}–{j}")
    ax_bot.set_xlabel("Applied force")
    ax_bot.set_ylabel(r"Bond stretch $r - r_0$")
    ax_bot.set_title("Bond stretch: weakest bond absorbs most deformation")
    ax_bot.legend(fontsize=7)
    ax_bot.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Interpreting the results

        **Force–extension curve:** The orange squares (quasi-static
        load-stepping) follow the stable branch up to the fold at
        $F_{\max} = 2.55$, where the LAMMPS minimiser crashes — the atom
        is lost. The blue circles (arclength continuation) trace smoothly
        through the fold onto the unstable branch (decreasing force).

        **Bond lengths:** Under increasing applied force, all bonds stretch,
        but the weakest bond ($D=0.85$) stretches the most. At the fold, it
        passes the inflection point $r^*$ and begins to snap — its length
        increases rapidly while the other bonds contract (since the total
        force is now decreasing). This confirms that **disorder determines
        which bond breaks**: the weakest link in the chain.

        In a 1D chain, once the first bond snaps the chain disconnects and
        subsequent bonds unload. For sequential multi-bond snapping (a
        staircase of folds), a 2D or 3D lattice with redundant load paths
        is needed — the kind of system LACT was designed for.
        """
    )
    return


if __name__ == "__main__":
    app.run()
