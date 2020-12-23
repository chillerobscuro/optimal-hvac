from datetime import timedelta
from typing import List, Tuple, Union

from pulp import PULP_CBC_CMD, LpMinimize, LpProblem, LpStatus, LpVariable, lpSum, value # type: ignore


class HVAC:
    def __init__(
        self,
        energy_cost: List[float],
        outdoor_temps: List[float],
        initial_building_temperature: float = 20,
        low_temp_threshold: float = 19.0,
        high_temp_threshold: float = 20.0,
        heat_mass_capacity: float = 1.65e7,
        maximum_power: float = 10e3,
        time_window_size: timedelta = timedelta(minutes=10),
        conditioned_floor_area: float = 100.0,
        heat_transmission: float = 200.0,
        verbose: bool = True,
        integer_opt_only: bool = False
    ) -> None:
        self.low_thresh = low_temp_threshold
        self.high_thresh = high_temp_threshold
        self.initial_temp = initial_building_temperature
        self.heat_mass_capacity = heat_mass_capacity
        self.heat_transmission = heat_transmission
        self.maximum_power = maximum_power
        self.current_temperature = initial_building_temperature
        self.time_window_size = time_window_size.total_seconds()
        self.conditioned_floor_area = conditioned_floor_area
        if len(energy_cost) != len(outdoor_temps):
            print("Energy cost and Outdoor Temps must have same number of values")
            return
        self.energy_cost = energy_cost
        self.outdoor_temps = outdoor_temps
        self.verbose = verbose
        self.integer_opt_only = integer_opt_only

    def next_temp(
        self, control_step: float, current_temp: float, outdoor_temp: float
    ) -> float:
        dt_by_cm = self.time_window_size / self.heat_mass_capacity
        next_temp = current_temp * (
            1 - dt_by_cm * self.heat_transmission
        ) + dt_by_cm * (
            control_step * self.maximum_power + self.heat_transmission * outdoor_temp
        )
        return next_temp

    def optimized_values(self) -> Tuple[List, List]:
        opt = self.optimize()
        return self.extract_values(opt)

    def optimize(self) -> LpProblem:
        ln = len(self.energy_cost)
        idx = list(range(ln))
        opt_prob = LpProblem("hvac", LpMinimize)
        cat = 'Integer' if self.integer_opt_only else "Continuous"
        control_steps = LpVariable.dicts("control_step", idx, cat=cat, lowBound=-1, upBound=1)
        abs_control_steps = LpVariable.dicts("abs_steps", idx)
        opt_prob += lpSum([self.energy_cost[i] * abs_control_steps[i] for i in idx])  # cost function

        temp_trackers = list(range(ln + 1))
        house_temp = LpVariable.dicts("house_temp", temp_trackers)
        house_temp[0] = self.initial_temp  # initial temperature of house - a known value

        # add variables to track house temp after each state
        for i in idx:
            # prob += house_temp[i + 1] == self.next_temp(i, control_steps, house_temp)
            opt_prob += house_temp[i + 1] == self.next_temp(control_steps[i], house_temp[i], self.outdoor_temps[i])
            opt_prob += house_temp[i + 1] <= self.high_thresh
            opt_prob += house_temp[i + 1] >= self.low_thresh

            # Constraints to set absolute value of W used
            opt_prob += abs_control_steps[i] >= control_steps[i]
            opt_prob += abs_control_steps[i] >= -control_steps[i]

        opt_prob.solve(PULP_CBC_CMD(msg=False))
        return opt_prob

    def extract_values(self, prob) -> Tuple[List, List]:
        obj = value(prob.objective)
        if self.verbose:
            print(f"Solution is {LpStatus[prob.status]}\nTotal cost of this regime is: {obj}")
        temps, states = [], []
        for v in prob.variables():
            if "house" in v.name:
                temps.append((int(v.name.split("_")[-1]), v.varValue))
            elif "control" in v.name:
                states.append((int(v.name.split("_")[-1]), v.varValue))

        # sort all optimization problem variables by integer in name, return values
        states.sort(key=lambda x: x[0])
        temps.sort(key=lambda x: x[0])
        state_values = [x[1] for x in states]
        temp_values = [x[1] for x in temps]
        return state_values, temp_values
