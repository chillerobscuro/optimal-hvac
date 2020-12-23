from hvac import HVAC
from typing import List, Tuple, Union
import numpy as np
import pandas as pd

import matplotlib  # type: ignore
matplotlib.use("agg")
import matplotlib.pyplot as plt  # type: ignore


class RunSim:
    def __init__(
        self,
        energy_cost: List[float],
        outdoor_temps: List[float],
        initial_building_temperature: float = 21.5,
        low_temp_threshold: float = 19.,
        high_temp_threshold: float = 22.,
        window_size: int = 96,
        verbose: bool = True,
        int_opt_only: bool = False,
        plot: bool = True,
        save_plot_loc: str = "images/test_plot.png"
    ) -> None:
        self.energy_cost = energy_cost
        self.outdoor_temps = outdoor_temps
        self.temperature = initial_building_temperature
        self.low_thresh = low_temp_threshold
        self.high_thresh = high_temp_threshold
        self.window_size = window_size if not window_size % 2 else window_size - 1  # must be even
        self.control_regime: List[float] = []
        self.house_temps: List[float] = []
        self.total_cost: float = 0.0
        self.opt_steps: int = 0
        self.plot = plot
        self.save_plot_loc = save_plot_loc
        self.verbose = verbose
        self.int_opt_only = int_opt_only

    def run(self) -> Tuple[List[float], List[float]]:
        self.opt_steps = 0
        ln: int = len(self.energy_cost)
        remainder = ln % self.window_size
        if remainder != 0:
            if self.verbose:
                print(f'window_size must be a factor of len(energy_cost), trimming arrays from {ln} to {ln-remainder}')
            self.energy_cost = self.energy_cost[:-remainder]
            self.outdoor_temps = self.outdoor_temps[:-remainder]
            ln -= remainder

        div_fac = 2  # step_size * div_fac = window_size
        step_size = int(self.window_size / div_fac)
        half_window = int(self.window_size / 2)
        for i in range(step_size, ln + 1, step_size):
            self.opt_steps += 1
            h = HVAC(
                self.energy_cost[i - half_window: i + half_window],
                self.outdoor_temps[i - half_window: i + half_window],
                low_temp_threshold=self.low_thresh,
                high_temp_threshold=self.high_thresh,
                initial_building_temperature=self.temperature,
                verbose=self.verbose,
                integer_opt_only=self.int_opt_only
            )
            states, temps = h.optimized_values()
            these_states = states[:step_size]
            self.control_regime.extend(these_states)
            self.house_temps.extend(temps[:step_size])
            self.temperature = temps[step_size - 1]

        # compute total cost of this control regime
        self.total_cost = sum(
            [abs(self.control_regime[x]) * self.energy_cost[x] for x in range(ln)]
        )

        if self.plot:
            self.plot_controls()

        return self.control_regime, self.house_temps

    def plot_controls(self) -> None:
        t = range(len(self.house_temps))
        fig, (ax1, ax3) = plt.subplots(nrows=2, sharex='all', figsize=(15, 8))

        color = 'black'
        ax1.set_ylabel('House temperature', color=color)
        ax1.plot(t, self.house_temps, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.axhline(self.high_thresh, ls="--")
        ax1.axhline(self.low_thresh, ls="--")
        ax1.title.set_text(f'Optimized Control by Energy Cost : {round(self.total_cost, 2)}')

        ax2 = ax1.twinx()
        color = 'blue'
        ax2.set_ylabel('Outdoor temperature', color=color)
        ax2.plot(t, self.outdoor_temps, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        color = 'green'
        ax3.set_ylabel('Energy Cost', color=color)
        ax3.plot(t, self.energy_cost, color=color)
        ax3.tick_params(axis='y', labelcolor=color)
        ax3.set_xlabel('time step')

        ax4 = ax3.twinx()
        color = 'black'
        ax4.set_ylabel('Control State', color=color)
        ax4.plot(t, self.control_regime, 'o', color=color, alpha=.5)
        ax4.tick_params(axis='y', labelcolor=color)
        ax4.axhline(0, ls="--")

        fig.tight_layout()
        plt.savefig(self.save_plot_loc)


if __name__ == "__main__":

    dff = pd.read_csv('data/test_data.csv')
    energies = dff['energies'].values
    outdoor_temp = dff['outdoor_temp'].values

    hs = RunSim(energies, outdoor_temp, int_opt_only=False, verbose=True)
    hs.run()
