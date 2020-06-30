import numpy as np
from tqdm import tqdm
from collections import namedtuple
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('dark')


class MonteCarloAnnealer:
    def __init__(self, config, seed=665):
        np.random.seed(seed)
        self.rng = np.random.RandomState(seed)
        self.config = config
        self.coordinates = {k: range(len(d)) for (k, d) in self.config.domain.items()}
        self.state = {}
        self.memory = {}
        self.trajectory_data = namedtuple('trajectory', 'state, cost, T, accept')
        self.potential = None

        self.random_start()
        self.reset_memory()
        self.visualize_potential()

        print('Domain dim: {:d}'.format(np.prod([x.stop for x in self.coordinates.values()])))

    def reset_memory(self):
        self.memory = {}

    def update_memory(self, step, state, cost, temperature, accept):
        self.memory.update({step: self.trajectory_data(state, cost, temperature, accept)})

    def update_state(self, new_state):
        assert isinstance(new_state, dict), "states must be dict"
        assert self.config.domain.keys() == set(new_state.keys()), "new states must have identical keys with domain"
        self.state = new_state

    def random_start(self):
        init_state = {}
        for k, domain_indxs in self.coordinates.items():
            init_state.update({k: self.rng.choice(domain_indxs)})
        self.update_state(init_state)
        # return init_state

    def _select_random_transition(self):
        transition = {}
        while True:
            for k in self.config.domain:
                transition.update({k: self.rng.choice([-1, 0, 1])})
            if list(transition.values()).count(0) != len(self.config.domain):
                break

        return transition

    def random_neighbor(self):
        neighbor_state = {}
        while True:
            transition = self._select_random_transition()
            for k, d in self.config.domain.items():
                neighbor_state.update({k: self.state[k] + transition[k]})

            if all(map(lambda x: x[0] in x[1], zip(neighbor_state.values(), self.coordinates.values()))):
                break

        return neighbor_state

    def compute_temperature(self, step):
        # step goes from 0 to k_max - 1
        t = self.config.initial_temperature * (self.config.max_steps - 1 - step) / (self.config.max_steps - 1) \
            + self.config.residual_temperature
        return t

    def accept(self, cost, new_cost, temp):
        p = np.exp(-(new_cost - cost) / temp)
        if p >= self.rng.rand():
            return True
        else:
            return False

    def compute_cost(self, state):
        v = ()
        for k, idx in state.items():
            v += (self.config.domain[k][idx],)

        w_components = self.config.w * sum(np.power(v, 6))
        gaussian_components = [
            np.prod(tuple(np.exp(-(v[k] - z[k]) ** 2 / (2 * self.config.sigma ** 2)) for k in range(len(z))))
            for z in self.config.minima_coordinates
        ]
        cost = w_components - sum(np.prod(list(zip(self.config.minima_depths, gaussian_components)), axis=-1))

        return cost

    def run_simulation(self):
        self.random_start()  # choose starting state randomly
        self.reset_memory()  # reset memory to record trajecotry data

        cost = float("inf")
        accept = True
        for step in tqdm(range(self.config.max_steps)):
            if accept:
                cost = self.compute_cost(self.state)
            temperature = self.compute_temperature(step)

            new_state = self.random_neighbor()
            new_cost = self.compute_cost(new_state)

            accept = self.accept(cost, new_cost, temperature)
            self.update_memory(step, self.state, cost, temperature, accept)

            if accept:
                self.update_state(new_state)

        print('[INFO] simulation done, final values:')
        msg = '[INFO] final state: {},  corresponds to: {},   cost: {:.3f}:'
        print(msg.format(
            self.state,
            tuple(np.round(self.config.domain[k][idx], 3) for (k, idx) in self.state.items()),
            self.compute_cost(self.state)
        ))

        costs_traj = [traj.cost for step, traj in self.memory.items()]
        plt.plot(costs_traj)
        plt.xlabel('time')
        plt.ylabel('cost')
        plt.grid()
        plt.show()

    def visualize_potential(self):
        domain_sizes = tuple(len(d) for d in self.config.domain.values())
        potential = np.zeros(domain_sizes)

        x_domain = self.config.domain['x']
        y_domain = self.config.domain['y']

        for i in range(domain_sizes[0]):
            for j in range(domain_sizes[1]):
                v = (x_domain[i], y_domain[j])
                potential[j, i] = self.config.w * sum(np.power(v, 6))
                gaussian_components = [
                    np.prod(tuple(np.exp(-(v[k] - z[k]) ** 2 / (2 * self.config.sigma ** 2)) for k in range(len(z))))
                    for z in self.config.minima_coordinates
                ]
                potential[j, i] -= sum(np.prod(list(zip(self.config.minima_depths, gaussian_components)), axis=-1))

        self.potential = potential

        xtick_labels = np.arange(np.min(x_domain), np.max(x_domain) + 1)
        xticks = [np.argmin(abs(item - x_domain)) for item in xtick_labels]

        ytick_labels = np.arange(np.min(y_domain), np.max(y_domain) + 1)
        yticks = [np.argmin(abs(item - y_domain)) for item in ytick_labels]

        plt.figure(figsize=(8, 6))
        plt.imshow(potential, cmap='RdBu')
        plt.xticks(xticks, xtick_labels)
        plt.yticks(yticks, ytick_labels)
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.title('Energy Landscape', fontsize=13)
        plt.show()
