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
    # Helper to create a fresh dimer system (called twice: once for
    # quasi-static-only, once for continuation)
    def make_dimer():
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

            group fixed id 1
            group mobile id 2
            fix freeze fixed setforce 0.0 0.0 0.0

            fix pull mobile addforce 0.0 0.0 0.0
            fix_modify pull energy yes

            compute forces all property/atom fx fy fz
            compute ids all property/atom id
            """
        )

        def update_command(force):
            return f"""
            unfix pull
            fix pull mobile addforce {force} 0.0 0.0
            fix_modify pull energy yes
            """

        return atom_cont_system(lmp, update_command)

    return (make_dimer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Quasi-static load-stepping (for comparison)

        First we try the naive approach: ramp the applied force in small steps
        from 0 to 3.0 using LAMMPS energy minimisation at each step.
        This works up to the fold ($F \approx 2.40$), but at the next step
        there is no energy minimum — the atom flies to the pair-potential
        cutoff and the solution is lost.
        """
    )
    return


@app.cell
def _(make_dimer, np):
    # Quasi-static only: force 0 → 3.0 in steps of 0.1
    qs_system = make_dimer()
    qs_system.quasi_static_run(0.0, 0.1, 30, verbose=False)

    qs_U0 = qs_system.U_0
    qs_seps = []
    qs_forces = []
    for _Y in qs_system.data["Y_s"]:
        _pos = _Y[:-1].reshape(-1, 3)
        _r = (qs_U0[1, 0] + _pos[1, 0]) - (qs_U0[0, 0] + _pos[0, 0])
        qs_seps.append(_r)
        qs_forces.append(_Y[-1])
    qs_seps = np.array(qs_seps)
    qs_forces = np.array(qs_forces)
    return qs_forces, qs_seps


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Arclength continuation

        Now we use LACT. A short quasi-static ramp seeds the path with a few
        points, then `continuation_run` takes over and traces smoothly through
        the fold and back to zero force on the unstable branch.
        """
    )
    return


@app.cell
def _(make_dimer, np):
    # Continuation system: seed with 5 quasi-static points, then continue
    cont_system = make_dimer()
    cont_system.quasi_static_run(0.0, 0.5, 5, verbose=True)
    n_qs_cont = len(cont_system.data["Y_s"])

    cont_system.continuation_run(
        n_iter=100,
        ds_default=0.1,
        ds_smallest=0.001,
        ds_largest=1.0,
        verbose=True,
        checkpoint_freq=0,
        cont_target=0.0,
        target_tol=0.05,
    )

    cont_U0 = cont_system.U_0
    cont_seps = []
    cont_forces = []
    for _Y in cont_system.data["Y_s"]:
        _pos = _Y[:-1].reshape(-1, 3)
        _r = (cont_U0[1, 0] + _pos[1, 0]) - (cont_U0[0, 0] + _pos[0, 0])
        cont_seps.append(_r)
        cont_forces.append(_Y[-1])
    cont_seps = np.array(cont_seps)
    cont_forces = np.array(cont_forces)
    return cont_forces, cont_seps, n_qs_cont


@app.cell
def _(np):
    # Analytical LJ force–extension curve
    # At equilibrium: F_applied = 24*(1/r^7 - 2/r^13) for sigma=1, eps=1
    r_anal = np.linspace(0.95, 2.5, 500)
    F_anal = 24.0 * (1.0 / r_anal**7 - 2.0 / r_anal**13)
    return F_anal, r_anal


@app.cell
def _(F_anal, cont_forces, cont_seps, n_qs_cont, plt, qs_forces, qs_seps, r_anal):
    fig, ax = plt.subplots(figsize=(7, 5))

    # Analytical reference
    ax.plot(r_anal, F_anal, "k-", lw=1, alpha=0.4, label="Analytical LJ")

    # Quasi-static (load-stepping) — shows failure past the fold
    ax.plot(qs_seps, qs_forces, "s", ms=5, color="C1", alpha=0.7,
            label="Quasi-static (load-stepping)")

    # Continuation — seed points + arclength
    ax.plot(cont_seps[:n_qs_cont], cont_forces[:n_qs_cont],
            "D", ms=5, color="C0", alpha=0.5)
    ax.plot(cont_seps[n_qs_cont:], cont_forces[n_qs_cont:],
            "o-", ms=3, lw=1.5, color="C0", label="Arclength continuation")

    # Mark the fold
    i_max = F_anal.argmax()
    ax.plot(r_anal[i_max], F_anal[i_max], "r*", ms=12, zorder=5,
            label="Fold (spinodal)")

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

        - **Load-stepping** (orange squares) fails: at $F > F_{\max}$ there
          is no energy minimum, so the LAMMPS minimiser pushes the atom all
          the way to the pair-potential cutoff. The solution jumps
          discontinuously and is lost.
        - **Arclength continuation** (blue circles) succeeds: it treats force
          and separation jointly, stepping along the solution curve by arc
          length. At the fold the force *decreases* while the separation
          *also decreases* (moving onto the repulsive/unstable branch).
          The continuation data traces the full S-curve, matching the
          analytical result exactly.
        """
    )
    return


if __name__ == "__main__":
    app.run()
