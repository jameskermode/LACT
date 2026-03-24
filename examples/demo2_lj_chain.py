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

    # --- LAMMPS setup ---
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

        # Morse potential with per-bond well depths.
        # Cutoff 5.0 keeps the bond force nonzero at large separation.
        pair_style morse 5.0
        pair_coeff * * 0.001 {alpha} {r0} 0.001
        pair_coeff 1 2 {bond_info[0][1]} {alpha} {r0} 5.0
        pair_coeff 2 3 {bond_info[1][1]} {alpha} {r0} 5.0
        pair_coeff 3 4 {bond_info[2][1]} {alpha} {r0} 5.0
        pair_coeff 4 5 {bond_info[3][1]} {alpha} {r0} 5.0
        pair_coeff 5 6 {bond_info[4][1]} {alpha} {r0} 5.0

        # Pin left end; pull right end with applied force
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

    chain = atom_cont_system(lmp, update_command)
    return alpha, bond_info, chain, n_atoms, r0


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Running the continuation

        1. **Quasi-static ramp** — force from 0 to 2.0 (below the first fold
           at $F_{\max} = 0.85 \times 6/2 = 2.55$).
        2. **Arclength continuation** — traces through the fold of the weakest
           bond and onto the unstable branch where the force drops.
        """
    )
    return


@app.cell
def _(chain):
    chain.quasi_static_run(0.0, 0.5, 5, verbose=True)

    # Trace through the fold and back down to force ≈ 1.0
    chain.continuation_run(
        n_iter=200,
        ds_default=0.1,
        ds_smallest=0.001,
        ds_largest=0.5,
        verbose=True,
        checkpoint_freq=0,
        cont_target=1.0,
        target_tol=0.1,
    )
    return


@app.cell
def _(alpha, bond_info, chain, n_atoms, np, r0):
    # Extract data at each solution point
    U_0 = chain.U_0

    extensions = []
    applied_forces = []
    bond_lengths_all = []

    for Y in chain.data["Y_s"]:
        dev = Y[:-1].reshape(n_atoms, 3)
        pos = U_0 + dev

        ext = pos[-1, 0] - U_0[-1, 0]
        extensions.append(ext)
        applied_forces.append(Y[-1])

        blens = [pos[i + 1, 0] - pos[i, 0] for i in range(n_atoms - 1)]
        bond_lengths_all.append(blens)

    extensions = np.array(extensions)
    applied_forces = np.array(applied_forces)
    bond_lengths_all = np.array(bond_lengths_all)

    # Theoretical fold forces per bond
    F_max_vals = [(ids, D, D * alpha / 2) for ids, D in bond_info]
    return F_max_vals, applied_forces, bond_lengths_all, extensions


@app.cell
def _(applied_forces, extensions, plt):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(extensions, applied_forces, "o-", ms=3, lw=1.5)
    ax.set_xlabel("Extension (displacement of atom 6)")
    ax.set_ylabel("Applied force")
    ax.set_title("Force–extension through the first fold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig
    return


@app.cell
def _(F_max_vals, alpha, applied_forces, bond_lengths_all, np, plt, r0):
    fig2, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    r_fold = r0 + np.log(2) / alpha

    for idx, ((i, j), D, Fmax) in enumerate(F_max_vals):
        ax_top.plot(applied_forces, bond_lengths_all[:, idx], lw=1.5,
                    label=f"Bond {i}–{j} ($D$={D:.2f}, $F_{{max}}$={Fmax:.2f})")

    ax_top.axhline(r_fold, color="k", ls="--", lw=0.8, alpha=0.5,
                   label=f"Fold at $r^*$={r_fold:.3f}")
    ax_top.set_ylabel(r"Bond length $r$")
    ax_top.set_title("Bond lengths vs. applied force")
    ax_top.legend(fontsize=7, loc="upper left")
    ax_top.grid(True, alpha=0.3)

    # Deviation from equilibrium
    for idx, ((i, j), D, _) in enumerate(F_max_vals):
        ax_bot.plot(applied_forces, bond_lengths_all[:, idx] - r0, lw=1.5,
                    label=f"Bond {i}–{j}")
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

        **Force–extension curve:** The S-curve shows the fold where the
        applied force reaches its maximum and the chain can no longer sustain
        higher load. Arclength continuation traces smoothly through the fold
        onto the unstable branch (decreasing force).

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
