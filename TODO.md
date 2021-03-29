# C++ implementation

- [X] check the simulation implementation.
- [X] complete algorithm.
- [X] add Mujoco and Envs.
- [ ] DB class for training.
- [ ] Trainig method in (controller/separate class).
- [ ] link testing with github to perform auto testing.

## Testing

 - [X] Testing software.
 - [X] verify implementation.
    - [X] cost.
    - [X] model.
    - [X] main_algo.
    - [X] decompose update function for simpler testing.
    - [X] test shif and init see also [#3](/../../issues/1)
 - [ ] Write argument checks and test them.

## Performance

  - [ ] profile the graph with [Profiler tool](https://www.tensorflow.org/guide/profiler)
  - [ ] Identify bottleneck and see how to improve performances.

## Code clearness

  - [ ] rewrite env class.

## Documentation

  - [ ] Write documentation.
  - [ ] Doxygene?


# Python Implementation

  - [X] add elements to replay buffer.
  - [X] finish training method.
  - [X] Rewrite Testing.
  - [X] log cost. (~1/2 day)
  - [X] extend dimensions. (~1/2 day)
  - [X] test extended dimension. (~1 day)
  - [X] generate tf_summary with hyperparam and env name. (~2 days)
    - [X] add config file parsing, to load hyperparameters and remove hard coding of the params.
    - [X] add argument parsing for the config file, env file etc.
    This will be usefull for organised testing.
  - [X] sort rendering problem.
  - [X] plot weighted simulated paths.
  - [X] look at noise distribution.
  - [X] log weights.
  - [X] log nabla in % of K.
  - [X] change the goal be a signal.
  - [X] look at signal filtering for control input.
  - [X] decoupling temprature from control cost.
  - [X] implement covariance decoupling between natural noise and artifical noise. Simple variance multiplier first.
  - [X] save task_config file.
  - [X] log inputs.
  - [X] log best cost.
  - [X] Sort tensorflow memory usage.
  - [X] normailze cost
  - [X] log normalized cost.
  - [X] debug flag when logging summary.
  - [ ] log predictied cost.
  - [X] train only if enough samples.
  - [X] moved logging to a separated pseudo singleton.
  - [X] log speed and position cost for elipse.
  - [ ] tune filtering process.
  - [ ] log disturbances.
  - [ ] add current. Needs to be parameterizable.
  - [ ] complexify mujoco simulation with friction and drag etc. (Need to find model for this.) (~(1-2) day)
  - [ ] learn more complex mujoco simulations. (~(2-3) days, need to learn the physics and derive the equations and  implement them)
  - [ ] Test Cpu. (probably inference on GPU, learning on CPU)
  - [ ] Train simple network on Point mass system.
  - [ ] Complexify learning (NN, GP etc). (~1-2 weeks) need to implement model, search the hyperparameters, same if we try Gaussian processes (GPflow) probably not a priority.
  - [ ] Experiment agent learning parameters. (dont know enough about the rest to make a prediction)
  - [ ] vehicle modelling.
  - [ ] go to uuv_sim.
  - [ ] uuv_sim on simulator.
  - [ ] Performance testing.
  - [ ] go to real robot

## Testing

  - [X] Test Model.
  - [X] Test Cost.
  - [X] Test main.
  - [X] Test shift and init.
  - [X] Test elips cost.
  - [X] Test gamma and upsilon.
