# RLLAB_intern
activities in RLLAB as an intern

- Isaac Sim
- Isaac SDK
- Reinforcement Learning



To Make Jackal extension & use jackal in ISAAC SIM :
1. Import "jackal.urdf" in Isaac Sim and save it as "jackal.usd"
2. Modify the location of your "jackal.usd" file in Jackal.py
3. Copy omni.isaac.jackal folder to isaac-sim/exts/
4. Add "omni.isaac.jackal" = {} in omni.isaac.sim.python.kit file, located in isaac-sim/apps
5. Open Isaac Sim and in Viewport -> Extensions, type omni.isaac.jackal and click the buttons to always include the extension



To Make Examples that can be opened via UI :
1. Make name_of_the_file_extension.py similar with hello_world_extension.py
2. Make name_of_the_file.py similar with hello_world.py
3. Add fun stuffs in name_of_the_file.py to make interesting examples!
