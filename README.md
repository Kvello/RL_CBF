# Quadrotor RL Policy
This repository contains the code used to develop, implement, and test a RL-policy for navigation of a quadrotor.
The method proposes including a penalty in the reward function based on the magnitude of the constraint violation

## :bulb: Main contributions
The main contributions of this work are under the task definition under
```bash
src/RL_CBF/task
```
and
```bash
RL_CBF/config/task_config
```
All project specific modification are in the
```bash
src/RL_CBF
```
subdirectory, except from minor modifications of both submodules.
Notably the aerialgym project was modified to allow for termination upon success, and the
CBF_Quadrotor subproject was modified to allow for simplified dynamics models, as well as some minor refactoring.
Lastly the plotting utilities as well as the data used for the plots are stored under
```bash
src/plotting
```
## :hammer_and_wrench: Setup and dependencies
This project dependes on a working [IsaacGym](https://junxnone.github.io/isaacgymdocs/index.html) installation with all its dependencies.
Following the guide at [AerialGym Documentation](https://ntnu-arl.github.io/aerial_gym_simulator/) is a good start.
Note however that this project uses a forked, and slightly modified version of aerialgym, found in the aerial_gym_simulator subdirectory.

The requirements are listed in the requirements.txt file. Simply run
```bash
pip install -r requirements.txt
```
preferably in a clean python environment. Python 3.8.20 was used.
[Pyenv](https://github.com/pyenv/pyenv) or [nix](https://nixos.org/) is highly recommended for setting up the environment.
Optionally any other (python) environment manager can be used.
To run the training simply run.
```bash
python runner.py --file=./ppo_CBF_quad_navigation.yaml --train --num_envs=[N] --headless=True
```
where N can be adjusted based on computational resources availible. 
## :weight_lifting_man: Training
To train the policies in simulation requires installing the aerialgym and CBF_Quadrotor subprojects as pip packages.
Run 
```bash
pip install -e .
```
in each of the 

## :chart_with_upwards_trend: Generating plots
To generate plots run the plotter.py script with arguments. E.g
```bash
python plotter.py --paths ../data/Benchmark ../data/CBF100 --set-names Benchmark CBF --plot-type steps epochs training-collisions successrate-timeseries
```
generates plots of the number of epochs and steps until the successrate of 0.95 has been reached,
as well as the number of collisions during training, and the time series of success rate vs. epochs across all runs grouped in the two 
categories "CBF" and "Benchmark". The --paths argument gives the paths to the specific group of files used for the plots.
This means that the number of paths specified must be equal to the number of set names.

The data in the
```bash
src/plotting/data/CBF-100
src/plotting/data/Benchmark
```
directories is the training data, while the data in the
```bash
src/plotting/data/eval-data
```
is the data of the trained policies evaluated in simulation over 256 episodes.

## :bar_chart: Results
Some of the results are shown bellow:

[successrate_timeseries.pdf](https://github.com/user-attachments/files/18185281/successrate_timeseries.pdf)
[training_collisions.pdf](https://github.com/user-attachments/files/18185285/training_collisions.pdf)
[training_epochs.pdf](https://github.com/user-attachments/files/18185288/training_epochs.pdf)
[crashrate_timeseries.pdf](https://github.com/user-attachments/files/18185289/crashrate_timeseries.pdf)
[cbf_constraint_timeseries.pdf](https://github.com/user-attachments/files/18185290/cbf_constraint_timeseries.pdf)
