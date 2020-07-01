import numpy as np
from matplotlib import animation, cm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('dark')


def create_animation(annealer, save_file=None):
    tot_steps = len(annealer.memory)
    cmap = cm.get_cmap("Greens")
    cmap_start_step = tot_steps / 4

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(28, 8))

    sc = ax[0].scatter(x=[], y=[], s=30, marker='.', cmap="Greens")
    im = ax[0].imshow(annealer.potential, cmap='RdBu')
    ax[0].set_xlabel('x')
    ax[0].set_xlabel('y')

    x_domain = annealer.config.domain['x']
    y_domain = annealer.config.domain['y']

    xtick_labels = np.arange(np.min(x_domain), np.max(x_domain) + 1)
    xticks = [np.argmin(abs(item - x_domain)) for item in xtick_labels]

    ytick_labels = np.arange(np.min(y_domain), np.max(y_domain) + 1)
    yticks = [np.argmin(abs(item - y_domain)) for item in ytick_labels]

    ax[0].set_xticks(xticks)
    ax[0].set_xticklabels(xtick_labels)
    ax[0].set_yticks(yticks)
    ax[0].set_yticklabels(ytick_labels)

    ax[0].invert_yaxis()
    plt.colorbar(im, ax=ax[0])
    ax[0].set_title('Energy Landscape', fontsize=16)

    all_costs = [annealer.memory[i].cost for i in range(tot_steps)]

    min_cost = min(all_costs)
    max_cost = max(all_costs)
    delta = max_cost - min_cost

    ln = ax[1].plot([], [])[0]
    ax[1].set_xlabel('time', fontsize=10)
    ax[1].set_xlim(0, tot_steps)
    ax[1].set_ylim(min_cost - delta / 5, max_cost + delta / 5)
    ax[1].set_title('Trajectory Costs', fontsize=16)
    plt.grid()

    msg = 'Initial T = {:d},   max_steps = {:d}'
    plt.suptitle(msg.format(annealer.config.initial_temperature, annealer.config.max_steps), fontsize=20)

    def update_fig(step):
        sc_data = np.array([tuple(annealer.memory[i].state.values()) for i in range(step + 1)])
        sc.set_offsets(sc_data)

        rgba = []
        for i in np.arange(cmap_start_step, cmap_start_step + len(sc_data)):
            f = np.minimum((i + 1) / tot_steps, 1.0)
            rgba.append(np.expand_dims(cmap(f), 0))
        rgba = np.concatenate(rgba)
        sc.set_color(rgba)

        ln_data = [annealer.memory[i].cost for i in range(step + 1)]
        ln.set_xdata(range(len(ln_data)))
        ln.set_ydata(ln_data)

        title = 'Trajectory Costs,  current temperature = {:.2f}'
        ax[1].set_title(title.format(annealer.memory[step].T), fontsize=16)

        return sc, ln

    ani = animation.FuncAnimation(fig, update_fig, tot_steps, interval=5, blit=True)
    writer = animation.writers['ffmpeg'](fps=240)

    dpi = 100
    if save_file is None:
        save_file = 'animation.mp4'
    ani.save(save_file, writer=writer, dpi=dpi)

    return ani
