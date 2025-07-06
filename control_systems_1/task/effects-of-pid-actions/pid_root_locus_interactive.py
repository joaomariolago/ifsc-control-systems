from __future__ import annotations
import io
import numpy as np
import control as ctrl
from control.pzmap import _find_root_locus_gain
import matplotlib.pyplot as plt
import imageio.v3 as iio

def create_pid(kp: float, ki: float = 0.0, kd: float = 0.0) -> ctrl.TransferFunction:
    """Retorna TF de um PID ideal Kp + Ki/s + Kd·s."""
    s = ctrl.TransferFunction.s
    return kp + ki / s + kd * s


def closed_loop(
    plant: ctrl.TransferFunction,
    kp: float,
    ki: float,
    kd: float,
) -> ctrl.TransferFunction:
    """Closed loop:  C(s)·P(s) / (1 + C(s)·P(s))"""
    return ctrl.feedback(create_pid(kp, ki, kd) * plant)


def interactive_pid_rootlocus(
    plant: ctrl.TransferFunction,
    *,
    kp0: float = 1.0,
    ki0: float = 0.0,
    kd0: float = 0.0,
    tune: str = "K",
    k_max: float = 500.0,
    n_points: int = 4000,
    gif_filename: str | None = None,
    gif_fps: int = 3,
) -> None:
    """
    Click on the root-locus → plots (step + Bode) updated in real time.
    If `gif_filename` is passed, generates a GIF with the chosen screens.
    Parameters:
        plant: The plant transfer function
        kp0: Initial proportional gain
        ki0: Initial integral gain
        kd0: Initial derivative gain
        tune: The type of gain to adjust ("K", "P", "I", "D")
        k_max: Maximum gain value for the root locus
    """

    tune = tune.upper()
    if tune not in {"K", "P", "I", "D"}:
        raise ValueError("tune must be 'K', 'P', 'I' or 'D'")

    # Base controller
    s = ctrl.TransferFunction.s
    C_base = create_pid(kp0, ki0, kd0)
    if tune == "K":
        L_term = C_base
    else:
        term   = {"P": 1, "I": 1/s, "D": s}[tune]
        fixed  = C_base - {"P": kp0, "I": ki0/s, "D": kd0*s}[tune]
        L_term = term
        C_base = fixed

    # Calculate locus for plotting
    gains = np.linspace(0, k_max, n_points)
    rlocus, _ = ctrl.root_locus(L_term * plant,
                                gains=gains, plot=False, grid=False)

    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    ax_rl,  ax_step = axs[0]
    ax_mag, ax_ph   = axs[1]

    # Title that displays the current PID gains
    pid_title = fig.suptitle(f"Kp = {kp0:.3g}   Ki = {ki0:.3g}   Kd = {kd0:.3g}",
                             y=0.99, fontsize=11, fontweight='bold')

    # root-locus
    for branch in rlocus.T:
        ax_rl.plot(branch.real, branch.imag, 'b')
    ax_rl.set(title=f"Root-Locus – clique ({tune})",
              xlabel="Re(s)", ylabel="Im(s)")
    ax_rl.grid()
    highlight, = ax_rl.plot([], [], 'ro', ms=7)

    # empty lines for the closed-loop plots
    step_line,  = ax_step.plot([], [], 'r')
    mag_line,   = ax_mag .plot([], [], 'r')
    phase_line, = ax_ph  .plot([], [], 'r')

    w = np.logspace(-2, 2, 800)
    ax_step.set(title="Step resp.", xlabel="t (s)", ylabel="y");        ax_step.grid()
    ax_mag .set(title="Bode |T(jω)|", xlabel="rad/s", ylabel="dB");    ax_mag.grid()
    ax_ph  .set(title="Bode ∠T(jω)",  xlabel="rad/s", ylabel="graus"); ax_ph .grid()

    # open-loop bode (dashed)
    mag_ol, ph_ol, _ = ctrl.bode_plot(L_term * plant, w, plot=False)
    ax_mag .semilogx(w, 20*np.log10(mag_ol), 'k--', label='OL')
    ax_ph  .semilogx(w, ph_ol*180/np.pi,     'k--', label='OL')
    ax_mag.legend(); ax_ph.legend()

    # List of frames for the GIF
    frames: list[np.ndarray] = []

    def _snapshot() -> None:
        """Captura a figura atual em RGB e empilha em `frames`."""
        if gif_filename is None:
            return
        fig.canvas.draw()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        buf.seek(0)
        frames.append(iio.imread(buf))

    def on_click(event):
        if event.inaxes is not ax_rl or event.xdata is None:
            return

        # Exact gain via internal helper
        K, s_click = _find_root_locus_gain(event, L_term*plant, ax_rl)
        if K is None or K < 0:
            return

        if tune == "K":
            kp, ki, kd = kp0*K, ki0*K, kd0*K
        elif tune == "P":
            kp, ki, kd = kp0 + K, ki0,      kd0
        elif tune == "I":
            kp, ki, kd = kp0,      ki0 + K, kd0
        else:
            kp, ki, kd = kp0,      ki0,      kd0 + K

        print(f"\nK = {K:.4g}  →  Kp = {kp:.4g}  Ki = {ki:.4g}  Kd = {kd:.4g}")

        # Update the title with the new gains
        pid_title.set_text(f"Kp = {kp:.3g}   Ki = {ki:.3g}   Kd = {kd:.3g}")

        T = closed_loop(plant, kp, ki, kd)

        # step response
        t, y = ctrl.step_response(T, T=np.linspace(0, 5, 1000))
        step_line.set_data(t, y)
        ax_step.relim(); ax_step.autoscale_view()

        # closed-loop bode
        mag_cl, ph_cl, _ = ctrl.bode_plot(T, w, plot=False)
        mag_line  .set_data(w, 20*np.log10(mag_cl))
        phase_line.set_data(w, ph_cl*180/np.pi)
        ax_mag .relim(); ax_mag .autoscale_view()
        ax_ph  .relim(); ax_ph  .autoscale_view()

        highlight.set_data([s_click.real], [s_click.imag])
        fig.canvas.draw_idle()

        # Saving for the gif
        _snapshot()

    fig.canvas.mpl_connect("button_release_event", on_click)

    def on_close(_event):
        if gif_filename is None or not frames:
            return
        iio.imwrite(gif_filename, np.stack(frames), fps=gif_fps)
        print(f"\nGIF saved: {gif_filename}  ({len(frames)} frames)")

    fig.canvas.mpl_connect("close_event", on_close)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    P = ctrl.TransferFunction([1], [1, 10, 20])
    interactive_pid_rootlocus(
        P,
        kp0=1.0, ki0=1, kd0=1,
        tune="K",
        k_max=100,
        gif_filename="assets/pid_tuning_k.gif",
        gif_fps=4,
    )
