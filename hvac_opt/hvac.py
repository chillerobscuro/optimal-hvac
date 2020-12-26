from datetime import timedelta
from typing import List, Tuple, Union

from pulp import PULP_CBC_CMD, LpMinimize, LpProblem, LpStatus, LpVariable, lpSum, value # type: ignore


class HVAC:
    def __init__(
        self,
        energy_cost: List[float],
        outdoor_temps: List[float],
        initial_building_temperature: float = 19.0,
        low_temp_threshold: float = 18.0,
        high_temp_threshold: float = 20.0,
        heat_mass_capacity: float = 1.65e7,
        maximum_power: float = 10e3,
        time_window_size: timedelta = timedelta(minutes=10),
        conditioned_floor_area: float = 100.0,
        heat_transmission: float = 200.0,
        verbose: bool = True,
        integer_opt_only: bool = False
    ) -> None:
        """
        Contains building parameters, temperature update function, and optimization functions
        energy_cost: List of forecasted energy cost at each time step. Can be dollar value, carbon emissions, etc
        outdoor_temps: List of forecasted outdoor temperatures (°C)
        initial_building_temperature: Initial indoor temp (°C)
        low_temp_threshold: Lowest allowable temperature
        high_temp_threshold: Highest allowable temperature
        heat_mass_capacity: (J/K)
        maximum_power: (W)
        time_window_size: timedelta corresponding to time step size of input arrays
        conditioned_floor_area: (m^2)
        heat_transmission: (W/K)
        verbose: Option to print
        integer_opt_only: Set to True if control values must be either [-1, 0, 1]
        """
        if len(energy_cost) != len(outdoor_temps):
            print("Energy cost and Outdoor Temps must have same number of values")
            return
        self.energy_cost = energy_cost
        self.outdoor_temps = outdoor_temps
        self.current_temperature = initial_building_temperature
        self.low_thresh = low_temp_threshold
        self.high_thresh = high_temp_threshold
        self.heat_mass_capacity = heat_mass_capacity
        self.maximum_power = maximum_power
        self.heat_transmission = heat_transmission
        self.time_window_size = time_window_size.total_seconds()
        self.conditioned_floor_area = conditioned_floor_area
        self.verbose = verbose
        self.integer_opt_only = integer_opt_only
        self.num_steps = len(energy_cost)

    def next_temp(
        self, control_step: float, current_temp: float, outdoor_temp: float
    ) -> float:
        """
        Computes the next indoor temperature based on the simple-simple building energy model by Tim Tröndle
        control_step: What wattage sent to the HVAC (negative cools, positive heats)
        current_temp: Current indoor temp
        outdoor_temp: Current outdoor temp
        return: Next temperature
        """
        dt_by_cm = self.time_window_size / self.heat_mass_capacity
        next_temp = current_temp * (
            1 - dt_by_cm * self.heat_transmission
        ) + dt_by_cm * (
            control_step * self.maximum_power + self.heat_transmission * outdoor_temp
        )
        self.current_temperature = next_temp
        return next_temp

    def optimized_values(self) -> Tuple[List, List]:
        """
        Run the optimizer and extract the values
        return: List of control actions, List of indoor temperatures
        """
        opt = self.optimize()
        return self.extract_values(opt)

    def optimize(self) -> LpProblem:
        """
        This function uses the PuLP library to find the optimal control regime
        return: The solved LpProblem object
        """
        idx = list(range(self.num_steps))
        opt_prob = LpProblem("hvac", LpMinimize)
        cat = 'Integer' if self.integer_opt_only else "Continuous"
        control_steps = LpVariable.dicts("control_step", idx, cat=cat, lowBound=-1, upBound=1)
        abs_control_steps = LpVariable.dicts("abs_steps", idx)
        opt_prob += lpSum([self.energy_cost[i] * abs_control_steps[i] for i in idx])  # cost function

        # Problem variable to track indoor temperature (depends on control outputs)
        temp_trackers = list(range(self.num_steps + 1))
        house_temp = LpVariable.dicts("house_temp", temp_trackers)
        house_temp[0] = self.current_temperature
        for i in idx:
            opt_prob += house_temp[i + 1] == self.next_temp(control_steps[i], house_temp[i], self.outdoor_temps[i])
            opt_prob += house_temp[i + 1] <= self.high_thresh
            opt_prob += house_temp[i + 1] >= self.low_thresh

            # Constraints to set absolute value of W used
            opt_prob += abs_control_steps[i] >= control_steps[i]
            opt_prob += abs_control_steps[i] >= -control_steps[i]

        opt_prob.solve(PULP_CBC_CMD(msg=False))
        return opt_prob

    def extract_values(self, prob: LpProblem) -> Tuple[List, List]:
        """
        Extract the values from the solved optimization object
        prob: The PuLP optimization object
        return: List of control actions, List of indoor temperatures
        """
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

    def on_off_controller(self, threshold_buffer: int = 0.5) -> Tuple[List, List]:
        """
        This control model outputs max cooling/heating wattage when the temperature gets within threshold_buffer of
        either temperature threshold.
        threshold_buffer: Degrees Celsius within threshold to turn HVAC on/off
        return: List of control actions, List of indoor temperatures
        """
        states, temps = list(), list()
        low_trigger, high_trigger = self.low_thresh + threshold_buffer, self.high_thresh - threshold_buffer
        for n in range(self.num_steps):
            t = self.current_temperature
            if t <= low_trigger:
                action = 1.
            elif t >= high_trigger:
                action = -1.
            else:
                action = 0
            states.append(action)
            self.next_temp(action, t, self.outdoor_temps[n])
            temps.append(self.current_temperature)
        return states, temps
