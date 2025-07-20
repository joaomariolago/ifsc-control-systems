import control
import numpy as np
import matplotlib.pyplot as plt
from control.pzmap import _find_root_locus_gain

#
# Definição do sistema
#
# G(s) = (s + 7) / (s(s + 5)(s + 15)(s + 20))
#

G = control.tf([1, 7], np.polymul([1, 0], np.polymul([1, 5], np.polymul([1, 15], [1, 20]))))

#
# Requisitos de desempenho:
#

overshoot_req = 5   # %
settle_time_req = 1 # segundos
wn_req = 1.8        # rad/s
zeta_req = 0.7

#
# Plot
#

# Limites do plot, pode se ajustar facilmente no plot interativo
initial_x_max_lim = 5     # Limite inicial do plot no eixo X máximo
initial_x_min_lim = -25   # Limite inicial do plot no eixo X mínimo
initial_y_max_lim = 2.5   # Limite inicial do plot no eixo Y máximo
initial_y_min_lim = -2.5  # Limite inicial do plot no eixo Y mínimo

fig, (ax_locus, ax_step) = plt.subplots(1, 2, figsize=(16, 6))

control.root_locus(G, ax=ax_locus, plot=True)
ax_locus.set_xlim([initial_x_min_lim, initial_x_max_lim])
ax_locus.set_ylim([initial_y_min_lim, initial_y_max_lim])
ax_locus.set_title("Root Locus")
ax_locus.set_xlabel("Eixo Real")
ax_locus.set_ylabel("Eixo Imaginário")
ax_locus.grid(True)

def plot_zeta_line(zeta, wn_max=100, label=True, **kwargs):
    """
    Plota uma linha de ζ constante no Root Locus.

    Args:
        zeta (float): Fator de amortecimento ζ.
        wn_max (float): Frequência natural não amortecida máxima.
        label (bool): Se True, adiciona uma label ao gráfico.
        **kwargs: Argumentos adicionais para a plotagem.
    """
    theta = np.arccos(zeta)
    x = np.linspace(0, -wn_max * np.cos(theta), 100)
    y = np.tan(theta) * x

    # Plotar as duas linhas (superior e inferior)
    ax_locus.plot(x, y, '--', color='gray', lw=1,
                  label=f'Requisito ζ = {zeta:.2f}', **kwargs)
    ax_locus.plot(x, -y, '--', color='gray', lw=1)

    if label:
        ax_locus.text(x[0], y[0], f'ζ = {zeta}', fontsize=8)

def plot_wn_circle(wn, **kwargs):
    """
    Plota um círculo de ωₙ constante no Root Locus.

    Args:
        wn (float): Frequência natural não amortecida.
        **kwargs: Argumentos adicionais para a plotagem.
    """
    theta = np.linspace(0, 2 * np.pi, 300)
    x = -wn * np.cos(theta)
    y = wn * np.sin(theta)

    ax_locus.plot(x, y, ':', color='gray', lw=1,
                  label=f'Requisito ωₙ = {wn:.2f} rad/s', **kwargs)

# Plotagem dos requisitos ωₙ e ζ
plot_zeta_line(zeta=zeta_req, wn_max=80)
plot_wn_circle(wn=wn_req)

# Habilita legenda e remove labels "sys[0]" existentes adicionadas pelo control.root_locus
handles, labels = ax_locus.get_legend_handles_labels()
filtered = [(h, l) for h, l in zip(handles, labels) if not l.startswith("sys")]
if filtered:
    handles, labels = zip(*filtered)
    ax_locus.legend(handles, labels, loc='lower left')

# Plotagem da resposta ao degrau
line_step, = ax_step.plot([], [], lw=2, label='Resposta')
vlines = []
hlines = []
text_metrics = None
ax_step.set_xlabel("Tempo (s)")
ax_step.set_ylabel("Amplitude")

def update_step_response(K):
    """
    Atualiza a resposta ao degrau para um ganho K específico.

    Args:
        K (float): Ganho do controlador.
    """
    global vlines, hlines, text_metrics

    system_cl = control.feedback(K * G, 1)
    t, y = control.step_response(system_cl)
    final_value = y[-1]

    # Apaga linhas anteriores
    for l in vlines + hlines:
        l.remove()
    vlines = []
    hlines = []

    # Apaga texto anterior
    if text_metrics:
        text_metrics.remove()
        text_metrics = None

    # Calcula métricas
    y_max = np.max(y)
    t_max = t[np.argmax(y)]
    overshoot = (y_max - final_value) / final_value * 100

    # Tempo de subida (10% a 90%) - 10% a 90% da resposta
    y_10 = 0.1 * final_value
    y_90 = 0.9 * final_value
    t_r_start = next((t[i] for i in range(len(y)) if y[i] >= y_10), None)
    t_r_end = next((t[i] for i in range(len(y)) if y[i] >= y_90), None)
    rise_time = t_r_end - t_r_start if t_r_start and t_r_end else None

    # Tempo de estabilização (permanência dentro de ±2% do valor final)
    tol = 0.02 * final_value
    t_settle = None
    for i in reversed(range(len(y))):
        if np.abs(y[i] - final_value) > tol:
            t_settle = t[i + 1] if i + 1 < len(t) else t[-1]
            break

    # Linha vertical para o tempo de estabilização
    if t_settle:
        vlines.append(ax_step.axvline(t_settle, color='blue', ls='-', label='T_estabilização (2%)'))

    # Atualiza linha de resposta
    line_step.set_data(t, y)
    ax_step.set_xlim([0, np.max(t)])
    ax_step.set_ylim([np.min(y)*0.9, np.max(y)*1.1])
    ax_step.set_title(f"Resposta ao Degrau com K = {K:.3f}")

    # Linhas horizontais - Valor final e amplitude máxima
    hlines.append(ax_step.axhline(final_value, color='black', ls='-', label='Valor final'))
    hlines.append(ax_step.axhline(y_max, color='red', ls='-', label='Amplitude máxima'))

    # Linhas verticais - Tempo máximo, início e fim da subida
    vlines.append(ax_step.axvline(t_max, color='purple', ls='-', label='t máximo'))
    if t_r_start:
        vlines.append(ax_step.axvline(t_r_start, color='green', ls='-', label='Início subida'))
    if t_r_end:
        vlines.append(ax_step.axvline(t_r_end, color='yellow', ls='-', label='Fim subida'))

    # Requisitos de overshoot (linha horizontal) - Overshoot máximo
    hlines.append(ax_step.axhline(final_value * (1 + overshoot_req / 100),
                                  color='gray', ls='--', label=f'Overshoot máx ({overshoot_req}%)'))

    # Requisito de tempo de subida (linha vertical) - Tempo de subida máximo
    vlines.append(ax_step.axvline(settle_time_req,
                                  color='gray', ls='--', label=f'T_estab máx ({settle_time_req:.1f}s)'))

    # Habilita legenda
    ax_step.legend(loc='lower right')

    # Adiciona texto com métricas
    text_metrics = ax_step.text(0.98, 0.3,
                f"Overshoot: {overshoot:.2f}%\n"
                f"T_subida: {rise_time:.2f}s\n"
                f"T_estab: {t_settle:.2f}s\n"
                f"Amplitude máx: {y_max:.2f}",
                transform=ax_step.transAxes,
                fontsize=10,
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle="round", facecolor='white', alpha=0.7))


    fig.canvas.draw_idle()

def _click_dispatcher(event):
    """
    Dispatcher interativo para o Root Locus.

    Args:
        event (matplotlib.backend_bases.MouseEvent): Evento de clique.
    """
    if event.inaxes != ax_locus:
        return
    try:
        K, _ = _find_root_locus_gain(event, G, ax_locus)
        update_step_response(K)
    except Exception as e:
        print(f"[Erro ao calcular K]: {e}")

# Conecta o dispatcher ao evento de clique
fig.canvas.mpl_connect('button_release_event', _click_dispatcher)
fig.suptitle("Root Locus Interativo com Requisitos", fontsize=14)

# Plota a resposta ao degrau para K = 1
update_step_response(K=1)
plt.tight_layout()
plt.show()
