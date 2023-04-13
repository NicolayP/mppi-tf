import tensorflow as tf
gpu = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(device=gpu[0], enable=True)

from ..src.models.model_base import ModelBase
from ..src.models.point_mass_model import PointMassModel
from ..src.models.auv_model import AUVModel
from ..src.models.nn_model import NNAUVModel, Predictor, VelPred
from ..src.misc.utile import dtype, npdtype

import numpy as np

def qmul(q1, q2):
    mul = []
    for p, q in zip(q1, q2):
        w0, x0, y0, z0 = p
        w1, x1, y1, z1 = q
        res = np.array([
                        [
                            w0*w1 - x0*x1 - y0*y1 - z0*z1,
                            w0*x1 + x0*w1 + y0*z1 - z0*y1,
                            w0*y1 - x0*z1 + y0*w1 + z0*x1,
                            w0*z1 + x0*y1 - y0*x1 + z0*w1
                        ]
                        ], dtype=npdtype)

        mul.append(res)

    mul = np.concatenate(mul, axis=0)
    return mul


class TestPointMassModel(tf.test.TestCase):
    def setUp(self):
        self.dt=0.1
        pass

    def test_step1_k1_s2_a1_m1(self):
        s = 2
        a = 1
        m = 1.
        model = PointMassModel(m, self.dt, s, a)

        state_in = np.array([[[0.], [0.]]], dtype=npdtype)
        action_in = np.array([[[1.]]], dtype=npdtype)

        acc = self.dt*self.dt/(2.*m)
        vel = self.dt/m
        exp_u = np.array([[[acc], [vel]]], dtype=npdtype)
        exp_x = np.array([[[0.], [0.]]], dtype=npdtype)
        exp = exp_u + exp_x

        x_pred = model.build_free_step_graph("", state_in)
        u_pred = model.build_action_step_graph("", action_in)
        pred = model.build_step_graph("", state_in, action_in)

        self.assertAllClose(exp_u, u_pred)
        self.assertAllClose(exp_x, x_pred)
        self.assertAllClose(exp, pred)

    def test_step1_k1_s4_a2_m1(self):
        s = 4
        a = 2
        m = 1.
        model = PointMassModel(m, self.dt, s, a)

        state_in = np.array([[[0.], [0.], [0.], [0.]]], dtype=npdtype)
        action_in = np.array([[[1.], [1.]]], dtype=npdtype)

        acc = self.dt*self.dt/(2.*m)
        vel = self.dt/m
        exp_u = np.array([[[acc], [vel], [acc], [vel]]], dtype=npdtype)
        exp_x = np.array([[[0.], [0.], [0.], [0.]]], dtype=npdtype)
        exp = exp_u + exp_x

        x_pred = model.build_free_step_graph("", state_in)
        u_pred = model.build_action_step_graph("", action_in)
        pred = model.build_step_graph("", state_in, action_in)

        self.assertAllClose(exp_u, u_pred)
        self.assertAllClose(exp_x, x_pred)
        self.assertAllClose(exp, pred)

    def test_step1_k5_s6_a3_m1d5(self):
        s = 6
        a = 3
        m = 1.5
        model = PointMassModel(m, self.dt, s, a)

        state_in = np.array([[[0.], [0.], [0.], [0.], [0.], [0.]],
                             [[2.], [1.], [5.], [0.], [-1.], [-2.]],
                             [[0.5], [0.5], [0.5], [0.5], [0.5], [0.5]],
                             [[1.], [0.], [1.], [0.], [1.], [0.]],
                             [[-1.], [0.5], [-3.], [2.], [0.], [0.]]], dtype=npdtype)
        action_in = np.array([[[1.], [1.], [1.]],
                              [[2.], [0.], [-1.]],
                              [[0.], [0.], [0.]],
                              [[0.5], [-0.5], [0.5]],
                              [[3.], [3.], [3.]]], dtype=npdtype)

        acc = self.dt*self.dt/(2.*m)
        vel = self.dt/m
        exp_u = np.array([[[acc], [vel], [acc], [vel], [acc], [vel]],
                          [[2.*acc], [2.*vel], [0.*acc], [0.*vel], [-1.*acc], [-1.*vel]],
                          [[0.*acc], [0.*vel], [0.*acc], [0.*vel], [0.*acc], [0.*vel]],
                          [[0.5*acc], [0.5*vel], [-0.5*acc], [-0.5*vel], [0.5*acc], [0.5*vel]],
                          [[3.*acc], [3.*vel], [3.*acc], [3.*vel], [3.*acc], [3.*vel]]], dtype=npdtype)

        exp_x = np.array([[[0.], [0.], [0.], [0.], [0.], [0.]],
                          [[2.+self.dt], [1.], [5.], [0.], [-1.-2.*self.dt], [-2.]],
                          [[0.5+0.5*self.dt], [0.5], [0.5+0.5*self.dt], [0.5], [0.5+0.5*self.dt], [0.5]],
                          [[1.], [0.], [1.], [0.], [1.], [0.]],
                          [[-1.+0.5*self.dt], [0.5], [-3.+2.*self.dt], [2.], [0.], [0.]]], dtype=npdtype)
        exp = exp_u + exp_x

        x_pred = model.build_free_step_graph("", state_in)
        u_pred = model.build_action_step_graph("", action_in)
        pred = model.build_step_graph("", state_in, action_in)

        self.assertAllClose(exp_u, u_pred)
        self.assertAllClose(exp_x, x_pred)
        self.assertAllClose(exp, pred)

    def test_init_k5_s6_a3_m1d5(self):
        s = 6
        a = 3
        m = 1.5
        model = PointMassModel(m, self.dt, s, a)

        state_in = np.array([[[-1.], [0.5], [-3.], [2.], [0.], [0.]]], dtype=npdtype)
        action_in = np.array([[[1.], [1.], [1.]],
                              [[2.], [0.], [-1.]],
                              [[0.], [0.], [0.]],
                              [[0.5], [-0.5], [0.5]],
                              [[3.], [3.], [3.]]], dtype=npdtype)

        acc = self.dt*self.dt/(2.*m)
        vel = self.dt/m
        exp_u = np.array([[[acc], [vel], [acc], [vel], [acc], [vel]],
                          [[2.*acc], [2.*vel], [0.*acc], [0.*vel], [-1.*acc], [-1.*vel]],
                          [[0.*acc], [0.*vel], [0.*acc], [0.*vel], [0.*acc], [0.*vel]],
                          [[0.5*acc], [0.5*vel], [-0.5*acc], [-0.5*vel], [0.5*acc], [0.5*vel]],
                          [[3.*acc], [3.*vel], [3.*acc], [3.*vel], [3.*acc], [3.*vel]]], dtype=npdtype)

        exp_x = np.array([[[-1+0.5*self.dt], [0.5], [-3.+2.*self.dt], [2.], [0.], [0.]],
                          [[-1+0.5*self.dt], [0.5], [-3.+2.*self.dt], [2.], [0.], [0.]],
                          [[-1+0.5*self.dt], [0.5], [-3.+2.*self.dt], [2.], [0.], [0.]],
                          [[-1+0.5*self.dt], [0.5], [-3.+2.*self.dt], [2.], [0.], [0.]],
                          [[-1+0.5*self.dt], [0.5], [-3.+2.*self.dt], [2.], [0.], [0.]]], dtype=npdtype)

        exp = exp_u + exp_x

        exp_x = np.expand_dims(exp_x[0, :, :], 0)

        x_pred = model.build_free_step_graph("", state_in)
        u_pred = model.build_action_step_graph("", action_in)
        pred = model.build_step_graph("", state_in, action_in)

        self.assertAllClose(exp_u, u_pred)
        self.assertAllClose(exp_x, x_pred)
        self.assertAllClose(exp, pred)

    def test_step3_k5_s6_a3_m1d5(self):
        s = 6
        a = 3
        m = 1.5
        model = PointMassModel(m, self.dt, s, a)

        state_in = np.array([[[0.], [0.], [0.], [0.], [0.], [0.]],
                             [[2.], [1.], [5.], [0.], [-1.], [-2.]],
                             [[0.5], [0.5], [0.5], [0.5], [0.5], [0.5]],
                             [[1.], [0.], [1.], [0.], [1.], [0.]],
                             [[-1.], [0.5], [-3.], [2.], [0.], [0.]]], dtype=npdtype)
        action_in = np.array([[[1.], [1.], [1.]],
                              [[2.], [0.], [-1.]],
                              [[0.], [0.], [0.]],
                              [[0.5], [-0.5], [0.5]],
                              [[3.], [3.], [3.]]], dtype=npdtype)

        acc = self.dt*self.dt/(2.*m)
        vel = self.dt/m

        # exp = A**3 * state_in + B*action_in * ( A**2 + A + I)

        # A**3 = | 1 self.dt**3 |
        #        | 0        1   |
        # A**2 = | 1 self.dt**2 |
        #        | 0        1   |

        B_u = np.array([[[1.0*3*(acc+vel*self.dt)], [1.0*vel*3], [1.0*3*(acc+vel*self.dt)], [1.0*vel*3], [1.0*3*(acc+vel*self.dt)], [1.0*vel*3]],
                        [[2.0*3*(acc+vel*self.dt)], [2.0*vel*3], [0.0*3*(acc+vel*self.dt)], [0.0*vel*3], [-1.0*3*(acc+vel*self.dt)], [-1.0*vel*3]],
                        [[0.0*3*(acc+vel*self.dt)], [0.0*vel*3], [0.0*3*(acc+vel*self.dt)], [0.0*vel*3], [0.0*3*(acc+vel*self.dt)], [0.0*vel*3]],
                        [[0.5*3*(acc+vel*self.dt)], [0.5*vel*3], [-0.5*3*(acc+vel*self.dt)], [-0.5*vel*3], [0.5*3*(acc+vel*self.dt)], [0.5*vel*3]],
                        [[3.0*3*(acc+vel*self.dt)], [3.0*vel*3], [3.0*3*(acc+vel*self.dt)], [3.0*vel*3], [3.0*3*(acc+vel*self.dt)], [3.0*vel*3]]], dtype=npdtype)

        exp_x = np.array([[[0.], [0.], [0.], [0.], [0.], [0.]],
                          [[2.+self.dt*3], [1.], [5.], [0.], [-1.-2.*self.dt*3], [-2.]],
                          [[0.5+0.5*self.dt*3], [0.5], [0.5+0.5*self.dt*3], [0.5], [0.5+0.5*self.dt*3], [0.5]],
                          [[1.], [0.], [1.], [0.], [1.], [0.]],
                          [[-1.+0.5*self.dt*3], [0.5], [-3.+2.*self.dt*3], [2.], [0.], [0.]]], dtype=npdtype)

        exp = B_u + exp_x

        pred = model.build_step_graph("", state_in, action_in)
        pred = model.build_step_graph("", pred, action_in)
        pred = model.build_step_graph("", pred, action_in)

        self.assertAllClose(exp, pred)

    def test_training(self):
        s = 4
        a = 2
        m = 1.5
        model = PointMassModel(m, self.dt, s, a)

        x = np.array([[0.5], [0.1], [0.2], [-0.1]], dtype=npdtype)
        u = np.array([[0.2], [-0.15]], dtype=npdtype)
        gt = np.array([[0.514], [0.14], [0.187], [-0.13]], dtype=npdtype)

        model.train_step(gt, x, u)

        mt = model.get_mass()

        self.assertAllLessEqual(mt, m)


class TestAUVModel(tf.test.TestCase):
    def setUp(self):
        self.params = dict()
        self.params["mass"] = 1000
        self.params["volume"] = 1.5
        self.params["density"] = 1000
        self.params["height"] = 1.6
        self.params["length"] = 2.5
        self.params["width"] = 1.5
        self.params["cog"] = [0, 0, 0]
        self.params["cob"] = [0, 0, 0.5]
        self.params["Ma"] = [[500., 0., 0., 0., 0., 0.],
                             [0., 500., 0., 0., 0., 0.],
                             [0., 0., 500., 0., 0., 0.],
                             [0., 0., 0., 500., 0., 0.],
                             [0., 0., 0., 0., 500., 0.],
                             [0., 0., 0., 0., 0., 500.]]
        self.params["linear_damping"] = [-70., -70., -700., -300., -300., -100.]
        self.params["quad_damping"] = [-740., -990., -1800., -670., -770., -520.]
        self.params["linear_damping_forward_speed"] = [1., 2., 3., 4., 5., 6.]
        self.inertial = dict()
        self.inertial["ixx"] = 650.0
        self.inertial["iyy"] = 750.0
        self.inertial["izz"] = 550.0
        self.inertial["ixy"] = 1.0
        self.inertial["ixz"] = 2.0
        self.inertial["iyz"] = 3.0
        self.params["inertial"] = self.inertial
        self.params["rk"] = 2
        self.model_quat = AUVModel({}, actionDim=6, dt=0.1, parameters=self.params)

    def test_B2I_transform_and_jacobian(self):
        ''' 
            Quat representation is [w, x, y, z] 
            quat: shape [k, 13, 1]
            euler representation is [roll, pitch, yaw]
            euler: shape [k, 12, 1]
        '''
        k = 3
        self.model_quat.set_k(k)

        quat = np.array([[[0.], [0.], [0.],
                          [0.], [0.], [0.], [1.],
                          [0.], [0.], [0.],
                          [0.], [0.], [0.]],
                         [[0.], [0.], [0.],
                           [0.0438308910967523], [0.25508068761447],
                           [0.171880267220619], [0.950510320581509],
                          [0.], [0.], [0.],
                          [0.], [0.], [0.]],
                         [[0.], [0.], [0.],
                           [-0.111618880991033], [0.633022223770408],
                           [0.492403876367579], [0.586824089619078],
                          [0.], [0.], [0.],
                          [0.], [0.], [0.]]], dtype=npdtype)

        pose_quat, speed_quat = self.model_quat.prepare_data(quat)

        # Quaternion section
        rot_from_lib = np.array([
                                 [
                                  [1.0000000, 0.0000000, 0.0000000],
                                  [0.0000000, 1.0000000, 0.0000000],
                                  [0.0000000, 0.0000000, 1.0000000]
                                 ],

                                 [
                                  [ 0.8107820, -0.3043871,  0.4999810],
                                  [ 0.3491088,  0.9370720,  0.0043632],
                                  [-0.4698463,  0.1710101,  0.8660254]
                                 ],

                                 [
                                  [-0.2863574, -0.7192234,  0.6330222],
                                  [ 0.4365945,  0.4901593,  0.7544065],
                                  [-0.8528685,  0.4924039,  0.1736482]
                                 ]
                                ], dtype=npdtype)

        wid=6
        xid=3
        yid=4
        zid=5

        exp_rot = np.array([
                            [
                             [
                              1.-2.*(quat[0, yid, 0]**2. + quat[0, zid, 0]**2.), 
                              2*(quat[0, xid, 0]*quat[0, yid, 0] - quat[0, zid, 0]*quat[0, wid, 0]),
                              2*(quat[0, xid, 0]*quat[0, zid, 0] + quat[0, yid, 0]*quat[0, wid, 0])
                             ],

                             [
                              2*(quat[0, xid, 0]*quat[0, yid, 0] + quat[0, zid, 0]*quat[0, wid, 0]),
                              1.-2.*(quat[0, xid, 0]**2. + quat[0, zid, 0]**2.),
                              2*(quat[0, yid, 0]*quat[0, zid, 0] - quat[0, xid, 0]*quat[0, wid, 0])
                             ],

                             [
                              2*(quat[0, xid, 0]*quat[0, zid, 0] - quat[0, yid, 0]*quat[0, wid, 0]),
                              2*(quat[0, yid, 0]*quat[0, zid, 0] + quat[0, xid, 0]*quat[0, wid, 0]),
                              1.-2.*(quat[0, xid, 0]**2. + quat[0, yid, 0]**2.)
                             ]
                            ],

                            [
                             [
                              1.-2.*(quat[1, yid, 0]**2. + quat[1, zid, 0]**2.),
                              2*(quat[1, xid, 0]*quat[1, yid, 0] - quat[1, zid, 0]*quat[1, wid, 0]),
                              2*(quat[1, xid, 0]*quat[1, zid, 0] + quat[1, yid, 0]*quat[1, wid, 0])
                             ],

                             [
                              2*(quat[1, xid, 0]*quat[1, yid, 0] + quat[1, zid, 0]*quat[1, wid, 0]),
                              1.-2.*(quat[1, xid, 0]**2. + quat[1, zid, 0]**2.),
                              2*(quat[1, yid, 0]*quat[1, zid, 0] - quat[1, xid, 0]*quat[1, wid, 0])
                             ],

                             [
                              2*(quat[1, xid, 0]*quat[1, zid, 0] - quat[1, yid, 0]*quat[1, wid, 0]),
                              2*(quat[1, yid, 0]*quat[1, zid, 0] + quat[1, xid, 0]*quat[1, wid, 0]),
                              1.-2.*(quat[1, xid, 0]**2. + quat[1, yid, 0]**2.)
                             ]
                            ],
                            
                            [
                             [
                              1.-2.*(quat[2, yid, 0]**2. + quat[2, zid, 0]**2.),
                              2*(quat[2, xid, 0]*quat[2, yid, 0] - quat[2, zid, 0]*quat[2, wid, 0]),
                              2*(quat[2, xid, 0]*quat[2, zid, 0] + quat[2, yid, 0]*quat[2, wid, 0])
                             ],

                             [
                              2*(quat[2, xid, 0]*quat[2, yid, 0] + quat[2, zid, 0]*quat[2, wid, 0]),
                              1.-2.*(quat[2, xid, 0]**2. + quat[2, zid, 0]**2.),
                              2*(quat[2, yid, 0]*quat[2, zid, 0] - quat[2, xid, 0]*quat[2, wid, 0])
                             ],

                             [
                              2*(quat[2, xid, 0]*quat[2, zid, 0] - quat[2, yid, 0]*quat[2, wid, 0]),
                              2*(quat[2, yid, 0]*quat[2, zid, 0] + quat[2, xid, 0]*quat[2, wid, 0]),
                              1.-2.*(quat[2, xid, 0]**2. + quat[2, yid, 0]**2.)
                             ]
                            ]
                           ], dtype=npdtype)

        exp_TB2Iquat = np.zeros(shape=(k, 4, 3), dtype=npdtype)

        exp_TB2Iquat[:, 3, 0] = -quat[:, xid, 0]
        exp_TB2Iquat[:, 3, 1] = -quat[:, yid, 0]
        exp_TB2Iquat[:, 3, 2] = -quat[:, zid, 0]

        exp_TB2Iquat[:, 0, 0] = quat[:, wid, 0]
        exp_TB2Iquat[:, 0, 1] = -quat[:, zid, 0]
        exp_TB2Iquat[:, 0, 2] = quat[:, yid, 0]

        exp_TB2Iquat[:, 1, 0] = quat[:, zid, 0]
        exp_TB2Iquat[:, 1, 1] = quat[:, wid, 0]
        exp_TB2Iquat[:, 1, 2] = -quat[:, xid, 0]

        exp_TB2Iquat[:, 2, 0] = -quat[:, yid, 0]
        exp_TB2Iquat[:, 2, 1] = quat[:, xid, 0]
        exp_TB2Iquat[:, 2, 2] = quat[:, wid, 0]
        
        exp_TB2Iquat = 0.5*exp_TB2Iquat

        rotBtoI, TBtoIquat = self.model_quat.body2inertial_transform(pose_quat)

        self.assertAllClose(rotBtoI, rot_from_lib)
        self.assertAllClose(rotBtoI, exp_rot)

        self.assertAllClose(TBtoIquat, exp_TB2Iquat)

        exp_jac_quat = np.zeros(shape=(k, 7, 6), dtype=npdtype)
        exp_jac_quat[:, 0:3, 0:3] = exp_rot
        exp_jac_quat[:, 3:7, 3:6] = exp_TB2Iquat
        jac_quat = self.model_quat.get_jacobian(rotBtoI, TBtoIquat)

        self.assertAllClose(jac_quat, exp_jac_quat)

    def test_restoring(self):
        k = 2
        self.model_quat.set_k(k)

        roll = np.array([45., 13., 280.], dtype=npdtype)*(np.pi/180.)
        pitch = np.array([0., 110., 50.], dtype=npdtype)*(np.pi/180.)
        yaw = np.array([0., 25., 325.], dtype=npdtype)*(np.pi/180.)
        pose = np.array([
                         [[1.0], [1.0], [1.0], [roll[0]], [pitch[0]], [yaw[0]]],
                         [[1.5], [2.3], [0.7], [roll[1]], [pitch[1]], [yaw[1]]],
                         [[5.2], [-2.], [1.7], [roll[2]], [pitch[2]], [yaw[2]]]
                        ], dtype=npdtype)

        pose = np.array([
                         [[1.0], [1.0], [1.0], [0.3826834], [0.], [0.], [0.9238795]],
                         [[1.5], [2.3], [0.7], [-0.1127657], [0.8086476], [0.0328141], [0.5764513]],
                         [[5.2], [-2.], [1.7], [-0.4582488], [0.4839407], [0.0503092], [0.7438269]]
                        ], dtype=npdtype)

        roll = np.array([13., 280.], dtype=npdtype)*(np.pi/180.)
        pitch = np.array([110., 50.], dtype=npdtype)*(np.pi/180.)
        yaw = np.array([25., 325.], dtype=npdtype)*(np.pi/180.)

        pose = np.array([
                         [[1.5], [2.3], [0.7], [-0.1127657], [0.8086476], [0.0328141], [0.5764513]],
                         [[5.2], [-2.], [1.7], [-0.4582488], [0.4839407], [0.0503092], [0.7438269]]
                        ], dtype=npdtype)

        rotItoB = np.zeros(shape=(k, 3, 3), dtype=npdtype)
        rotBtoI = np.zeros(shape=(k, 3, 3), dtype=npdtype)
        
        for i in range(k):
            rotBtoI[i, :, :] = np.array([[
                                         np.cos(yaw[i])*np.cos(pitch[i]),
                                         -np.sin(yaw[i])*np.cos(roll[i])+np.cos(yaw[i])*np.sin(pitch[i])*np.sin(roll[i]),
                                         np.sin(yaw[i])*np.sin(roll[i])+np.cos(yaw[i])*np.cos(roll[i])*np.sin(pitch[i])
                                        ],
                                        [
                                         np.sin(yaw[i])*np.cos(pitch[i]),
                                         np.cos(yaw[i])*np.cos(roll[i])+np.sin(roll[i])*np.sin(pitch[i])*np.sin(yaw[i]),
                                         -np.cos(yaw[i])*np.sin(roll[i])+np.sin(pitch[i])*np.sin(yaw[i])*np.cos(roll[i])
                                        ],
                                        [
                                         -np.sin(pitch[i]),
                                         np.cos(pitch[i])*np.sin(roll[i]),
                                         np.cos(pitch[i])*np.cos(roll[i])
                                        ]]
                                        , dtype=npdtype)
            rotItoB[i, :, :] = rotBtoI[i, :, :].T
        
        W = self.model_quat._mass*self.model_quat._gravity
        B = self.model_quat._volume*self.model_quat._density*self.model_quat._gravity

        r_g = self.model_quat._cog
        r_b = self.model_quat._cob

        fng = np.array([0.0, 0.0, -W], dtype=npdtype)
        fnb = np.array([0.0, 0.0, B], dtype=npdtype)

        fbg = np.dot(rotItoB, fng)
        fbb = np.dot(rotItoB, fnb)

        mbg = np.cross(r_g, fbg)
        mbb = np.cross(r_b, fbb)

        exp_rest = np.zeros((k, 6, 1), dtype=npdtype)
        fb = -(fbb+fbg)
        mb = -(mbb+mbg)
        exp_rest[:, 0, 0] = fb[:, 0]
        exp_rest[:, 1, 0] = fb[:, 1]
        exp_rest[:, 2, 0] = fb[:, 2]

        exp_rest[:, 3, 0] = mb[:, 0]
        exp_rest[:, 4, 0] = mb[:, 1]
        exp_rest[:, 5, 0] = mb[:, 2]

        rotBtoI, TBtoIquat = self.model_quat.body2inertial_transform(pose)

        rest = self.model_quat.restoring_forces("rest", rotBtoI)

        self.assertAllClose(rotBtoI, rotBtoI)
        self.assertAllClose(rest, exp_rest)

    def test_damping(self):
        k = 3
        self.model_quat.set_k(k)
        vel = np.array([
                        [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0]],
                        [[2.0], [1.5], [1.0], [3.0], [3.5], [2.5]],
                        [[-2.0], [-1.5], [-1.0], [-3.0], [-3.5], [-2.5]]
                       ], dtype=npdtype)
        d = self.model_quat.damping_matrix("damp", vel)
        exp_damp = np.zeros(shape=(vel.shape[0], 6, 6), dtype=npdtype)
        for i in range(vel.shape[0]):
            exp_damp[i, :, :] = -1* np.diag(self.params["linear_damping"]) - vel[i, 0, 0]*np.diag(self.params["linear_damping_forward_speed"])

        exp_damp += np.expand_dims(-1*np.diag(self.params['quad_damping']), axis=0) * np.abs(vel)
        self.assertAllClose(d, exp_damp)

    def test_corrolis(self):
        k = 1
        self.model_quat.set_k(k)
        vel = np.array([[[1.0], [1.0], [1.0], [0.0], [0.0], [0.0]]], dtype=npdtype)
        Iv = [self.inertial["ixx"]*vel[0, 3, 0] - self.inertial["ixy"]*vel[0, 4, 0] - self.inertial["ixz"]*vel[0, 5, 0], 
              - self.inertial["ixy"]*vel[0, 3, 0] + self.inertial["iyy"]*vel[0, 4, 0] - self.inertial["iyz"]*vel[0, 5, 0],
              - self.inertial["ixz"]*vel[0, 3, 0] - self.inertial["iyz"]*vel[0, 4, 0] + self.inertial["izz"]*vel[0, 5, 0]]
        Mav = - np.dot(np.array(self.params["Ma"], dtype=npdtype), vel)

        crb = np.array([[[0., 0., 0., 0., self.params["mass"]*vel[0, 2, 0], -self.params["mass"]*vel[0, 1, 0]],
                        [0., 0., 0., -self.params["mass"]*vel[0, 2, 0], 0., self.params["mass"]*vel[0, 0, 0]],
                        [0., 0., 0., self.params["mass"]*vel[0, 1, 0], -self.params["mass"]*vel[0, 0, 0], 0.],
                        [0., self.params["mass"]*vel[0, 2, 0], -self.params["mass"]*vel[0, 1, 0], 0., Iv[2], -Iv[1]],
                        [-self.params["mass"]*vel[0, 2, 0], 0., self.params["mass"]*vel[0, 0, 0], -Iv[2], 0., Iv[0]],
                        [self.params["mass"]*vel[0, 1, 0], -self.params["mass"]*vel[0, 0, 0], 0., Iv[1], -Iv[0], 0.]]], dtype=npdtype)

        ca = np.array([[[0., 0., 0., 0., -Mav[2, 0, 0], Mav[1, 0, 0]],
                       [0., 0., 0., Mav[2, 0, 0], 0., -Mav[0, 0, 0]],
                       [0., 0., 0., -Mav[1, 0, 0], Mav[0, 0, 0], 0.],
                       [0., -Mav[2, 0, 0], Mav[1, 0, 0], 0., -Mav[5, 0, 0], Mav[4, 0, 0]],
                       [Mav[2, 0, 0], 0., -Mav[0, 0, 0], Mav[5, 0, 0], 0., -Mav[3, 0, 0]],
                       [-Mav[1, 0, 0], Mav[0, 0, 0], 0., -Mav[4, 0, 0], Mav[3, 0, 0], 0.]]], dtype=npdtype)

        c = self.model_quat.coriolis_matrix("coriolis", vel)

        self.assertAllClose(c, crb + ca)

    def test_step1_k1_s13_a6(self):
        k = 1
        self.model_quat.set_k(k)
        state = np.array([[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0],
                           [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]], dtype=npdtype)
        action = np.array([[[1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]], dtype=npdtype)
        next_state = self.model_quat.build_step_graph("step", state, action)

        # Two different shapes can be fed to the AUV network.
        state = np.array([[[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0],
                           [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]]], dtype=npdtype)
        action = np.array([[[[1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]]], dtype=npdtype)
        next_state_ext = self.model_quat.build_step_graph("step", state, action)

        self.assertAllClose(next_state, next_state_ext)
        pass

    def test_step1_k5_s12_a6(self):
        k = 5
        self.model_quat.set_k(k)
        state = np.array([ [ [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0],
                             [0.0], [0.0], [0.0], [0.0], [0.0], [0.0] ],

                           [ [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [1.0],
                             [0.1], [2.0], [2.0], [1.0], [2.0], [3.0] ],

                           [ [0.0], [2.0], [1.0], [0.2], [0.3], [0.0], [1.0],
                             [-1.], [-1.], [-1.], [-1.], [-1.], [-1.] ],

                           [ [5.0], [0.2], [0.0], [1.2], [0.0], [3.1], [1.0],
                             [0.0], [0.0], [0.0], [0.0], [0.0], [0.0] ],

                           [ [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0],
                             [1.0], [1.0], [1.0], [1.0], [1.0], [1.0] ] ], dtype=npdtype)

        action = np.array([[ [1.0], [1.0], [1.0], [1.0], [1.0], [1.0] ],
                           [ [1.0], [1.0], [1.0], [1.0], [1.0], [1.0] ],
                           [ [-1.], [-1.], [-1.], [-1.], [-1.], [-1.] ],
                           [ [2.0], [2.0], [2.0], [2.0], [2.0], [2.0] ],
                           [ [-1.], [-1.], [-1.], [-1.], [-1.], [-1.] ]], dtype=npdtype)

        next_state = self.model_quat.build_step_graph("step", state, action)


        state = np.array([ [[ [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0],
                             [0.0], [0.0], [0.0], [0.0], [0.0], [0.0] ]],

                           [[ [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [1.0],
                             [0.1], [2.0], [2.0], [1.0], [2.0], [3.0] ]],

                           [[ [0.0], [2.0], [1.0], [0.2], [0.3], [0.0], [1.0],
                             [-1.], [-1.], [-1.], [-1.], [-1.], [-1.] ]],

                           [[ [5.0], [0.2], [0.0], [1.2], [0.0], [3.1], [1.0],
                             [0.0], [0.0], [0.0], [0.0], [0.0], [0.0] ]],

                           [[ [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0],
                             [1.0], [1.0], [1.0], [1.0], [1.0], [1.0] ]] ], dtype=npdtype)

        action = np.array([[[ [1.0], [1.0], [1.0], [1.0], [1.0], [1.0] ]],
                           [[ [1.0], [1.0], [1.0], [1.0], [1.0], [1.0] ]],
                           [[ [-1.], [-1.], [-1.], [-1.], [-1.], [-1.] ]],
                           [[ [2.0], [2.0], [2.0], [2.0], [2.0], [2.0] ]],
                           [[ [-1.], [-1.], [-1.], [-1.], [-1.], [-1.] ]]], dtype=npdtype)

        next_state_ext = self.model_quat.build_step_graph("step", state, action)

        self.assertAllClose(next_state, next_state_ext)
        pass

'''
class TestNNAUVModel(tf.test.TestCase):
    def setUp(self):
        self.nn_auv = NNAUVModel()

    def testPrepareTrainingDataN1(self):
        state_t = np.array([
                            [
                             [1.0], [1.0], [0.5], # position
                             [0.0], [0.0], [0.0], [1.0], # Quaternion
                             [1.0], [0.0], [0.25], # Linear velocities
                             [0.0] ,[0.0] ,[0.0] # Angular velocities
                            ]
                           ], dtype=npdtype)

        state_t1 = np.array([
                             [
                              [2.0], [1.0], [0.75], # position
                              [0.0], [0.0], [0.0], [1.0], # Quaternion
                              [3.0], [3.5], [4.5], # Linear velocities
                              [5.5], [6.5], [7.5] # Angular velocities
                             ]
                            ], dtype=npdtype)

        action = np.array([
                           [
                            [1.0], [2.0], [3.0], [4.0], [5.0], [6.0]
                           ]
                          ], dtype=npdtype)


        (X, y) = self.nn_auv.prepare_training_data(state_t, state_t1, action)

        # Quaternion, velocities and action
        exp_x = np.array([[[0.0], [0.0], [0.0], [1.0], # Quaternion
                           [1.0], [0.0], [0.25], # Linear velocities
                           [0.0], [0.0], [0.0], #Angular velocities
                           [1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]], dtype=npdtype) # Action

        # DeltaT. Atm it is a difference between states so negative quaterion.
        exp_y = np.array([[[1.0], [0.0], [0.25], # Next position
                           [0.0], [0.0], [0.0], [0.0], # Quaternion
                           [2.0], [3.5], [4.25], # Linear velocities
                           [5.5], [6.5], [7.5]]], dtype=npdtype) # angular velocitiess

        self.assertAllClose(X, exp_x)
        self.assertAllClose(y, exp_y)

    def testPrepareDataN1(self):
        state = np.array([[
                           [0.0], [1.0], [2.0], # Position
                           [3.0], [4.0], [5.0], [6.0], # Quaternion
                           [7.0], [8.0], [9.0], # Linera velocities
                           [10.0], [11.0], [12.0] # Angular velocities
                         ]], dtype=npdtype)
        action = np.array([[
                            [13.0], [14.0], [15.0], # Forces
                            [16.0], [17.0], [18.0] # Torques
                           ]], dtype=npdtype)
        inp_data = self.nn_auv.prepare_data(state, action)
        exp_inp_data = np.array([[3.0, 4.0, 5.0, 6.0, # Quaternion
                                  7.0, 8.0, 9.0, # Linear velocities
                                  10.0, 11.0, 12.0, # Angular velocities
                                  13.0, 14.0, 15.0, # Forces
                                  16.0, 17.0, 18.0]], dtype=npdtype) # Torques

        self.assertAllClose(exp_inp_data, inp_data)

    def testPrepareDataN6(self):
        state = np.array([
                          [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0], [11.0], [12.0]],
                          [[18.0], [17.0], [16.0], [15.0], [14.0], [13.0], [12.0], [11.0], [10.0], [9.0], [8.0], [7.0], [6.0]],
                          [[-0.0], [-1.0], [-2.0], [-3.0], [-4.0], [-5.0], [-6.0], [-7.0], [-8.0], [-9.0], [-10.0], [-11.0], [-12.0]],
                          [[-18.0], [-17.0], [-16.0], [-15.0], [-14.0], [-13.0], [-12.0], [-11.0], [-10.0], [-9.0], [-8.0], [-7.0], [-6.0]],
                          [[-0.5], [-1.5], [-2.5], [-3.5], [-4.5], [-5.5], [-6.5], [-7.5], [-8.5], [-9.5], [-10.5], [-11.5], [-12.5]],
                          [[0.5], [1.5], [2.5], [3.5], [4.5], [5.5], [6.5], [7.5], [8.5], [9.5], [10.5], [11.5], [12.5]],
                         ], dtype=npdtype)
        action = np.array([
                           [[13.0], [14.0], [15.0], [16.0], [17.0], [18.0]],
                           [[5.0], [4.0], [3.0], [2.0], [1.0], [0.0]],
                           [[-13.0], [-14.0], [-15.0], [-16.0], [-17.0], [-18.0]],
                           [[-5.0], [-4.0], [-3.0], [-2.0], [-1.0], [-0.0]],
                           [[-13.5], [-14.5], [-15.5], [-16.5], [-17.5], [-18.5]],
                           [[13.5], [14.5], [15.5], [16.5], [17.5], [18.5]],
                          ], dtype=npdtype)

        inp_data = self.nn_auv.prepare_data(state, action)
        exp_inp_data = np.array([
                                 [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
                                 [15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
                                 [-3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0, -13.0, -14.0, -15.0, -16.0, -17.0, -18.0],
                                 [-15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, -0.0],
                                 [-3.5, -4.5, -5.5, -6.5, -7.5, -8.5, -9.5, -10.5, -11.5, -12.5, -13.5, -14.5, -15.5, -16.5, -17.5, -18.5],
                                 [3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5]
                                ], dtype=npdtype)
        self.assertAllClose(exp_inp_data, inp_data)


class TestVelPred(tf.test.TestCase):
    def setUp(self):
        self.in_sizes_state = np.arange(2, 125, 5)
        self.in_sizes_action = np.arange(2, 125, 5)
        self.topologies = [
            [32],
            [64, 32],
            [32, 64, 32],
            [64, 64, 64, 64],
            [128, 128, 128, 128],
            [64, 64, 64, 64, 64]
        ]

    def testInSizeAndTopologies(self):
        for s, a in zip(self.in_sizes_state, self.in_sizes_action):
            for t in self.topologies:
                vel_pred = VelPred(s, t)
                dummy_state = tf.random.normal(shape=(1, s), dtype=dtype)
                dummy_input = tf.random.normal(shape=(1, a), dtype=dtype)
                vel_pred = VelPred(s+a, t)
                out_vel = vel_pred.forward(dummy_state, dummy_input)
                gt_out = np.zeros(shape=(1, 6), dtype=npdtype)

                self.assertShapeEqual(gt_out, out_vel)

    def testInSizeAndTopologiesBatch(self):
        k = 64
        for s, a in zip(self.in_sizes_state, self.in_sizes_action):
            for t in self.topologies:
                vel_pred = VelPred(s, t)
                dummy_state = tf.random.normal(shape=(k, s), dtype=dtype)
                dummy_input = tf.random.normal(shape=(k, a), dtype=dtype)
                vel_pred = VelPred(s+a, t)
                out_vel = vel_pred.forward(dummy_state, dummy_input)
                gt_out = np.zeros(shape=(k, 6), dtype=npdtype)

                self.assertShapeEqual(gt_out, out_vel)


class TestPredictor(tf.test.TestCase):
    def setUp(self):
        self.in_size_state = 18
        self.in_size_act = 6
        self.topo = [64, 64]
        self.dt = 0.1
        self.h = 1
        in_size = self.h*(self.in_size_state-3+self.in_size_act)
        self.vel_pred = VelPred(in_size, self.topo)
        self.pred = Predictor(self.vel_pred, self.dt, h=self.h)

    def testSingleStep(self):
        dummy_state = tf.random.normal(shape=(1, self.h, self.in_size_state), dtype=dtype)
        dummy_act = tf.random.normal(shape=(1, self.h, self.in_size_act), dtype=dtype)

        next_state = self.pred.forward(dummy_state, dummy_act)
        gt_shape = np.zeros(shape=(1, self.in_size_state))

        self.assertShapeEqual(gt_shape, next_state)

    def testBatchStep(self):
        k = 64
        dummy_state = tf.random.normal(shape=(k, self.h, self.in_size_state), dtype=dtype)
        dummy_act = tf.random.normal(shape=(k, self.h, self.in_size_act), dtype=dtype)

        next_state = self.pred.forward(dummy_state, dummy_act)
        gt_shape = np.zeros(shape=(k, self.in_size_state), dtype=npdtype)

        self.assertShapeEqual(gt_shape, next_state)
'''