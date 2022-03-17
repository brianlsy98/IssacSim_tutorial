# RLLAB_intern
activities in RLLAB as an intern

- Isaac Sim
- Isaac SDK
- Reinforcement Learning



To Make Jackal extension & use jackal in ISAAC SIM:
0.0. Import "jackal.urdf" in Isaac Sim and save it as "jackal.usd"
0.1. Modify the location of your "jackal.usd" file in Jackal.py
1. Copy omni.isaac.jackal folder to isaac-sim/exts/
2. Add "omni.isaac.jackal" = {} in omni.isaac.sim.python.kit file, located in isaac-sim/apps
3. Open Isaac Sim and in Viewport -> Extensions, type omni.isaac.jackal and click the buttons to always include the extension
