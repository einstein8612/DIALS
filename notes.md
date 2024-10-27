the project runs a set of local simulators that each mimic 1 intersection, they then get checked based on the global simulator.

global_traffic.py
local_traffic.py

their stuff is duplicated so code maintenance is super annoying.

The idea is to create a new environment that uses openstreetmap data to generate the intersections and roads.


https://github.com/flow-project/flow/blob/master/tutorials/tutorial01_sumo.ipynb
https://github.com/flow-project/flow/blob/master/tutorials/tutorial06_osm.ipynb

A cool idea to explore is, to see how distributed the local simulators should be. Per intersection? Group of intersections? Whole city?
Need to find a method to split up an osm file into smaller parts.


== MEETING STUFF ==
- for the longest time did not understand the results of the simulation, since they looked super odd -> hard to divide intersections in openstreetmap