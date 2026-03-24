import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Demo 1: LJ Dimer — Tracing a Force–Extension Fold

        Two Lennard-Jones atoms are pulled apart by an applied force.
        The force–extension curve has a **fold** (spinodal) at the inflection
        point of the LJ potential ($r \approx 1.245\,\sigma$). Beyond this
        point, no static equilibrium exists under force control — the bond
        snaps. Standard load-stepping fails here because there is no energy
        minimum to converge to.

        **Arclength continuation** parameterises the solution path by its
        arc length rather than by the applied force, letting it smoothly
        traverse the fold and trace the full S-curve onto the unstable
        (repulsive) branch.
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
    # --- LAMMPS setup: 2 LJ atoms in a non-periodic box ---
    lmp = lammps(cmdargs=["-screen", "none"])
    lmp.commands_string(
        """
        units         lj
        dimension     3
        boundary      f f f
        atom_style    atomic
        atom_modify   map yes

        region box block -50.0 50.0 -50.0 50.0 -50.0 50.0
        create_box 1 box
        mass 1 1.0

        create_atoms 1 single 0.0 0.0 0.0
        create_atoms 1 single 1.12246 0.0 0.0
        reset_atom_ids sort yes

        pair_style lj/cut 3.0
        pair_coeff 1 1 1.0 1.0 3.0

        # Fix atom 1 at the origin
        group fixed id 1
        group mobile id 2
        fix freeze fixed setforce 0.0 0.0 0.0

        # Applied pulling force on atom 2 (initially zero)
        fix pull mobile addforce 0.0 0.0 0.0
        fix_modify pull energy yes

        # Computes required by LACT
        compute forces all property/atom fx fy fz
        compute ids all property/atom id
        """
    )

    # The continuation parameter is the applied force.
    # fix_modify energy yes is needed so LAMMPS minimize accounts for
    # the work done by the applied force.
    def update_command(force):
        return f"""
        unfix pull
        fix pull mobile addforce {force} 0.0 0.0
        fix_modify pull energy yes
        """

    system = atom_cont_system(lmp, update_command)
    return (system,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Running the continuation

        1. **Quasi-static ramp** — increase the applied force in small steps
           from 0 to 2.0 (well below the fold at $F_{\max}\approx 2.40$).
           LAMMPS energy minimisation finds the equilibrium at each step.
        2. **Arclength continuation** — starting from the last quasi-static
           point, the LACT solver traces the equilibrium path through the fold
           and back to zero force on the unstable branch.
        """
    )
    return


@app.cell
def _(system):
    # Quasi-static: ramp force from 0 to 2.0 in 5 steps
    system.quasi_static_run(0.0, 0.5, 5, verbose=True)

    # Continuation: trace through the fold and back to zero force
    system.continuation_run(
        n_iter=100,
        ds_default=0.1,
        ds_smallest=0.001,
        ds_largest=1.0,
        verbose=True,
        checkpoint_freq=0,
        cont_target=0.0,
        target_tol=0.05,
    )
    return


@app.cell
def _(np, system):
    # Extract separation and force at each solution point
    U_0 = system.U_0  # reference positions (natoms, 3)

    separations = []
    forces = []
    for Y in system.data["Y_s"]:
        pos = Y[:-1].reshape(-1, 3)
        r = (U_0[1, 0] + pos[1, 0]) - (U_0[0, 0] + pos[0, 0])
        separations.append(r)
        forces.append(Y[-1])

    separations = np.array(separations)
    forces = np.array(forces)

    # Analytical LJ force–extension curve
    # At equilibrium: F_applied = -F_LJ = 24*(1/r^7 - 2/r^13) for sigma=1, eps=1
    r_anal = np.linspace(0.95, 2.5, 500)
    F_anal = 24.0 * (1.0 / r_anal**7 - 2.0 / r_anal**13)
    return F_anal, forces, r_anal, separations


@app.cell
def _(F_anal, forces, plt, r_anal, separations):
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(r_anal, F_anal, "k-", lw=1, alpha=0.5, label="Analytical LJ")
    ax.plot(separations, forces, "o-", ms=3, lw=1.5, label="LACT continuation")

    # Mark the fold (maximum of analytical curve)
    i_max = F_anal.argmax()
    ax.plot(r_anal[i_max], F_anal[i_max], "r*", ms=12, zorder=5, label="Fold (spinodal)")

    ax.set_xlabel(r"Separation $r / \sigma$")
    ax.set_ylabel(r"Applied force $F$")
    ax.set_title("Force–extension curve: LJ dimer pull")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## What happened at the fold?

        The analytical curve (black) shows that the applied force required to
        hold two LJ atoms at separation $r$ has a **maximum** at the inflection
        point of the potential ($r^* \approx 1.245\,\sigma$,
        $F_{\max} \approx 2.40\,\varepsilon/\sigma$). Beyond this point:

        - **Load-stepping** would fail: there is no energy minimum at
          $F > F_{\max}$, so LAMMPS minimisation diverges.
        - **Arclength continuation** succeeds: it treats force and separation
          jointly, stepping along the solution curve by arc length. At the
          fold the force *decreases* while the separation *also decreases*
          (moving onto the repulsive/unstable branch). The LACT data (blue)
          traces the full S-curve, matching the analytical result.
        """
    )
    return


if __name__ == "__main__":
    app.run()
