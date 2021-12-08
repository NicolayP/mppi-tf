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
    - [X] Model base.
    - [X] Cost base.
    - [ ] Controller base.
    - [ ] auv_model.
    - [X] point mass model.
    - [X] nn model.
    - [ ] static cost.
    - [X] elipse cost.
 - [ ] Doxygene?


# Python Implementation

## implementations:

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
  - [X] log predicted cost.
  - [X] train only if enough samples.
  - [X] moved logging to a separated pseudo singleton.
  - [X] log speed and position cost for elipse.
  - [X] log arguments in the graphs dir to repeate experiments.
  - [X] write experiment bash script.
  - [X] write repeate experience from log script.
  - [X] write model abstract class.
  - [ ] write documentation.
  - [ ] perform argument checking.
  - [ ] tune filtering process.
  - [ ] log disturbances.
  - [ ] add current. Needs to be parameterizable.
  - [ ] Test Cpu. (probably inference on GPU, learning on CPU)
  - [ ] Train simple network on Point mass system.
  - [ ] Complexify learning (NN, GP etc). (~1-2 weeks) need to implement model, search the hyperparameters, same if we try Gaussian processes (GPflow) probably not a priority.
  - [ ] Experiment agent learning parameters. (dont know enough about the rest to make a prediction)

### NN_AUV implementation:
  Main Idea: We don't want to predict in the inertial frame because this would lead to a lack of generalization when far from the origin in unexplored state space regions. So the dataset is as follos: A state is defined by: [pose_I_Bt.T, vel_Bt_Bt.T].T where pose_I_Bt is the pose of the body at time t expressed in inertial frame and vel_Bt_Bt is the velocity of the body frame as time t expressed in the body frame at time t. For 2 states, s_{t}, s_{t+1} as well as the input action a_{t} the network training data is as follows: (X, y) \simeq ([rot_I_Bt.T, vel_Bt_Bt.T, a_Bt_Bt.T].T, [pose_Bt_Bt1.t, vel_Bt1_Bt1.T].T). Where pose_Bt_Bt1 is the pose of the body at time t+1 expressin in the body frame at time t.
  - [X] Batch transform.
  - [X] Batch invsere.
  - [X] Batch quaternion inverse.
  - [X] Batch pure quaternion.
  - [X] Batch rotate vector.
  - [X] Batch quaternion multiplication.
  - [X] Data Preparation.
  - [X] Training Data Preparation
  - [X] Prediction to Inertial Frame.
  - [ ] Create network.
  - [ ] Training routine.
  - [ ] Collect training data from gazebo.
  - [ ] Train simple network on the training data.
  

## ROS Integration

  - [x] run a tf-graph in ros with python3.
  - [X] adapt static cost and eliptic cost to 3d problems.
  - [X] vehicle modelling.
    - [X] step function.
    - [X] compute acceleration.
    - [X] compute damping.
    - [X] compute restoring forces.
    - [X] compute coriolis matrix.
    - [X] load ros model parameters.
      - [X] namespace.
      - [X] world frame.
      - [X] body frame.
      - [X] inertial.
      - [X] mass.
      - [X] volume.
      - [X] density.
      - [X] height.
      - [X] length.
      - [X] width.
      - [X] center of gravity.
      - [X] center of buoyancy.
      - [X] added mass.
      - [X] gravity?
      - [X] linear damping.
      - [X] linear damping forward speed.
      - [X] quadratic damping.
      - [X] build mass matrix
    - [X] test step function.
    - [X] Why is model exploding (sign error in the restoring forces).
      - [X] Damping matrix computation had an error. Done, compared with the UUV implementation and have no diff anymore.
      - [X] Rerun the data gen process and see if it still explodes. It still explodes. Need to look at the angles as they are not normalized.
      - [X] Implement quaternions. 
      - [X] Test quaternions.
        - [X] check rot.
        - [X] check t.
        - [X] verify order of multiplication.
        - [X] check jacobian.
        - [X] check multiplication in step.
        - [X] check converstion.
    - [ ] look for learning mechanism.
    - [X] write test functions.
    - [ ] change variables to tensorflow variables.
    - [X] find solution to skew operation concat .

  - [ ] Performance testing.
  - [ ] go to real robot.

## Troubleshooting:

  - [ ] check why the acceleration in x is positive while the yaw angle is 180 and the input is 100N. The x acceleration should be negative.

## Testing

  - [X] Test Model.
  - [X] Test Cost.
  - [X] Test main.
  - [X] Test shift and init.
  - [X] Test elips cost.
  - [X] Test gamma and upsilon.

