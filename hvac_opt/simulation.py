import os
from pathlib import Path
from typing import List, Tuple, Union

import matplotlib  # type: ignore
matplotlib.use("agg")
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd

from hvac import HVAC

base_path = Path(os.path.abspath(os.path.dirname(__file__))).parent


class RunSim:
    def __init__(
        self,
        energy_cost: List[float],
        outdoor_temps: List[float],
        initial_building_temperature: float = 19.0,
        low_temp_threshold: float = 18.0,
        high_temp_threshold: float = 20.0,
        maximum_power: float = 10e3,
        window_size: int = 48,
        verbose: bool = True,
        int_opt_only: bool = False,
        plot: bool = True,
        save_plot_loc: Union[Path, str] = base_path / "images/test_plot.png"
    ) -> None:
        """
        Run the optimization algorithms stepwise and compute aggregate stats
        energy_cost: List of forecasted energy cost at each time step. Can be dollar value, carbon emissions, etc
        outdoor_temps: List of forecasted outdoor temperatures (°C)
        initial_building_temperature: Initial indoor temp (°C)
        low_temp_threshold: Lowest allowable temperature
        high_temp_threshold: Highest allowable temperature
        window_size: How many values to optimize each step
        verbose: Option to print
        int_opt_only: Set to True if control values must be either [-1, 0, 1]
        plot: Option to output plot
        save_plot_loc: Where to save plot
        """
        self.energy_cost = energy_cost
        self.outdoor_temps = outdoor_temps
        self.temperature = initial_building_temperature
        self.low_thresh = low_temp_threshold
        self.high_thresh = high_temp_threshold
        self.maximum_power = maximum_power,
        self.window_size = window_size if not window_size % 2 else window_size - 1  # must be even
        self.control_regime: List[float] = []
        self.house_temps: List[float] = []
        self.total_cost: float = 0.0
        self.opt_steps: int = 0
        self.plot = plot
        self.save_plot_loc = save_plot_loc
        self.verbose = verbose
        self.int_opt_only = int_opt_only
        self.num_steps = len(energy_cost)

    def run(self) -> Tuple[List[float], List[float]]:
        """
        Run the optimization algorithms stepwise and compute aggregate stats
        return: List of control actions, List of indoor temperatures
        """
        self.opt_steps = 0
        ln: int = self.num_steps
        remainder = ln % self.window_size
        if remainder != 0:
            if self.verbose:
                print(f'window_size must be a factor of len(energy_cost), trimming arrays from {ln} to {ln-remainder}')
            self.energy_cost = self.energy_cost[:-remainder]
            self.outdoor_temps = self.outdoor_temps[:-remainder]
            ln -= remainder
            self.num_steps -= remainder

        step_size = int(self.window_size / 2)
        for i in range(step_size, ln + 1, step_size):
            self.opt_steps += 1
            h = HVAC(
                self.energy_cost[i - step_size: i + step_size],
                self.outdoor_temps[i - step_size: i + step_size],
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

        self.compute_regime_cost()

        if self.plot:
            self.plot_controls('Optimized Control by Energy Cost :')

        return self.control_regime, self.house_temps

    def compute_regime_cost(self) -> None:
        """
        Compute total cost of this control regime
        """
        self.total_cost = sum(
            [abs(self.control_regime[x]*self.maximum_power) * self.energy_cost[x] for x in range(self.num_steps)]
        )

    def run_on_off(self):
        """
        Run simple model with binary control
        """
        h = HVAC(
            self.energy_cost,
            self.outdoor_temps,
            low_temp_threshold=self.low_thresh,
            high_temp_threshold=self.high_thresh,
            initial_building_temperature=self.temperature,
            verbose=self.verbose,
            integer_opt_only=self.int_opt_only
        )
        states, temps = h.on_off_controller()
        self.control_regime = states
        self.house_temps = temps

        self.compute_regime_cost()

        if self.plot:
            self.plot_controls('On/Off Control: ')

        return self.control_regime, self.house_temps

    def plot_controls(self, title: str) -> None:
        """
        Plot all values and save to path save_plot_loc
        """
        t = range(len(self.house_temps))
        fig, (ax1, ax3) = plt.subplots(nrows=2, sharex='all', figsize=(15, 8))

        color = 'black'
        ax1.set_ylabel('House temperature', color=color)
        ax1.plot(t, self.house_temps, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.axhline(self.high_thresh, ls="--")
        ax1.axhline(self.low_thresh, ls="--")
        ax1.title.set_text(f'{title} {round(self.total_cost, 2)}')

        ax2 = ax1.twinx()
        color = 'green'
        ax2.set_ylabel('Control State', color=color)
        ax2.plot(t, self.control_regime, 'o', color=color, alpha=.5)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.axhline(0, ls="--")

        color = 'black'
        ax3.set_ylabel('Energy Cost', color=color)
        ax3.plot(t, self.energy_cost, color=color)
        ax3.tick_params(axis='y', labelcolor=color)
        ax3.set_xlabel('time step')

        ax4 = ax3.twinx()
        color = 'blue'
        ax4.set_ylabel('Outdoor temperature', color=color)
        ax4.plot(t, self.outdoor_temps, color=color)
        ax4.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        plt.savefig(self.save_plot_loc)


def run_sims(opt_path: str = 'images/optimized.png', on_off_path: str = 'images/on_off.png'):
    """
    Run the Linear Optimization and on/off control, save plots for each
    """
    dff = pd.read_csv(base_path / 'data/test_data.csv')
    energy_cost = dff['energy_cost'].values
    outdoor_temp = dff['outdoor_temp'].values

    hs = RunSim(energy_cost, outdoor_temp, verbose=False, save_plot_loc=base_path / opt_path)
    hs.run()

    hs = RunSim(energy_cost, outdoor_temp, verbose=False, save_plot_loc=base_path / on_off_path)
    hs.run_on_off()


if __name__ == "__main__":
    run_sims()
