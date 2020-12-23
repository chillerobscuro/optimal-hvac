## Optimal HVAC Control for Buildings

This model finds the optimal control routine for heating and cooling a building based on the variable energy cost from
the grid. The cost of energy can be monetary, or a quantified carbon impact as demonstrated
[this paper](https://www.watttime.org/app/uploads/2019/03/Optimal-Refrigeration-Control-For-Soda-Vending-Machines_May_2015.pdf)
by Zoltan DeWitt and Matthew Roeschke. Data on your grid's marginal emmissions rate can be found using the
[WattTime API.](https://www.watttime.org/api-documentation/#introduction)

The building model is based on [this open source simple energy model](https://github.com/timtroendle/simple-simple) by
Tim Tröndle, based on the paywalled ISO 13790 standards.

Control is continuous by default, with -1.0 being max wattage towards cooling, and 1.0 as max heating. For integer or
binary programming, set `RunSim` class `int_opt_only` parameter to True.

To run simulation with test data, run `python hvac_opt/simulation.py` from home directory.

Run tests using `python -m pytest`