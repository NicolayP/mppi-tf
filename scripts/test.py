import tensorflow as tf
from model_base import ModelBase
from point_mass_model import PointMassModel
from auv_model import AUVModel
from cost_base import CostBase
from static_cost import StaticCost
from elipse_cost import ElipseCost
from controller_base import ControllerBase
import numpy as np
from uuv_control_msgs import srv

from quaternion import from_euler_angles

class TestPointMassModel(tf.test.TestCase):
    def setUp(self):
        self.dt=0.1
        pass

    def test_step1_k1_s2_a1_m1(self):
        s = 2
        a = 1
        m = 1.
        model = PointMassModel(m, self.dt, s, a)

        state_in = np.array([[[0.], [0.]]])
        action_in = np.array([[[1.]]])

        acc = self.dt*self.dt/(2.*m)
        vel = self.dt/m
        exp_u = np.array([[[acc], [vel]]])
        exp_x = np.array([[[0.], [0.]]])
        exp = exp_u + exp_x

        x_pred = model.buildFreeStepGraph("", state_in)
        u_pred = model.buildActionStepGraph("", action_in)
        pred = model.buildStepGraph("", state_in, action_in)

        self.assertAllClose(exp_u, u_pred)
        self.assertAllClose(exp_x, x_pred)
        self.assertAllClose(exp, pred)

    def test_step1_k1_s4_a2_m1(self):
        s = 4
        a = 2
        m = 1.
        model = PointMassModel(m, self.dt, s, a)

        state_in = np.array([[[0.], [0.], [0.], [0.]]])
        action_in = np.array([[[1.], [1.]]])

        acc = self.dt*self.dt/(2.*m)
        vel = self.dt/m
        exp_u = np.array([[[acc], [vel], [acc], [vel]]])
        exp_x = np.array([[[0.], [0.], [0.], [0.]]])
        exp = exp_u + exp_x

        x_pred = model.buildFreeStepGraph("", state_in)
        u_pred = model.buildActionStepGraph("", action_in)
        pred = model.buildStepGraph("", state_in, action_in)

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
                             [[-1.], [0.5], [-3.], [2.], [0.], [0.]]])
        action_in = np.array([[[1.], [1.], [1.]],
                              [[2.], [0.], [-1.]],
                              [[0.], [0.], [0.]],
                              [[0.5], [-0.5], [0.5]],
                              [[3.], [3.], [3.]]])

        acc = self.dt*self.dt/(2.*m)
        vel = self.dt/m
        exp_u = np.array([[[acc], [vel], [acc], [vel], [acc], [vel]],
                          [[2.*acc], [2.*vel], [0.*acc], [0.*vel], [-1.*acc], [-1.*vel]],
                          [[0.*acc], [0.*vel], [0.*acc], [0.*vel], [0.*acc], [0.*vel]],
                          [[0.5*acc], [0.5*vel], [-0.5*acc], [-0.5*vel], [0.5*acc], [0.5*vel]],
                          [[3.*acc], [3.*vel], [3.*acc], [3.*vel], [3.*acc], [3.*vel]]])

        exp_x = np.array([[[0.], [0.], [0.], [0.], [0.], [0.]],
                          [[2.+self.dt], [1.], [5.], [0.], [-1.-2.*self.dt], [-2.]],
                          [[0.5+0.5*self.dt], [0.5], [0.5+0.5*self.dt], [0.5], [0.5+0.5*self.dt], [0.5]],
                          [[1.], [0.], [1.], [0.], [1.], [0.]],
                          [[-1.+0.5*self.dt], [0.5], [-3.+2.*self.dt], [2.], [0.], [0.]]])
        exp = exp_u + exp_x

        x_pred = model.buildFreeStepGraph("", state_in)
        u_pred = model.buildActionStepGraph("", action_in)
        pred = model.buildStepGraph("", state_in, action_in)

        self.assertAllClose(exp_u, u_pred)
        self.assertAllClose(exp_x, x_pred)
        self.assertAllClose(exp, pred)

    def test_init_k5_s6_a3_m1d5(self):
        s = 6
        a = 3
        m = 1.5
        model = PointMassModel(m, self.dt, s, a)

        state_in = np.array([[[-1.], [0.5], [-3.], [2.], [0.], [0.]]])
        action_in = np.array([[[1.], [1.], [1.]],
                              [[2.], [0.], [-1.]],
                              [[0.], [0.], [0.]],
                              [[0.5], [-0.5], [0.5]],
                              [[3.], [3.], [3.]]])

        acc = self.dt*self.dt/(2.*m)
        vel = self.dt/m
        exp_u = np.array([[[acc], [vel], [acc], [vel], [acc], [vel]],
                          [[2.*acc], [2.*vel], [0.*acc], [0.*vel], [-1.*acc], [-1.*vel]],
                          [[0.*acc], [0.*vel], [0.*acc], [0.*vel], [0.*acc], [0.*vel]],
                          [[0.5*acc], [0.5*vel], [-0.5*acc], [-0.5*vel], [0.5*acc], [0.5*vel]],
                          [[3.*acc], [3.*vel], [3.*acc], [3.*vel], [3.*acc], [3.*vel]]])

        exp_x = np.array([[[-1+0.5*self.dt], [0.5], [-3.+2.*self.dt], [2.], [0.], [0.]],
                          [[-1+0.5*self.dt], [0.5], [-3.+2.*self.dt], [2.], [0.], [0.]],
                          [[-1+0.5*self.dt], [0.5], [-3.+2.*self.dt], [2.], [0.], [0.]],
                          [[-1+0.5*self.dt], [0.5], [-3.+2.*self.dt], [2.], [0.], [0.]],
                          [[-1+0.5*self.dt], [0.5], [-3.+2.*self.dt], [2.], [0.], [0.]]])

        exp = exp_u + exp_x

        exp_x = np.expand_dims(exp_x[0, :, :], 0)

        x_pred = model.buildFreeStepGraph("", state_in)
        u_pred = model.buildActionStepGraph("", action_in)
        pred = model.buildStepGraph("", state_in, action_in)

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
                             [[-1.], [0.5], [-3.], [2.], [0.], [0.]]])
        action_in = np.array([[[1.], [1.], [1.]],
                              [[2.], [0.], [-1.]],
                              [[0.], [0.], [0.]],
                              [[0.5], [-0.5], [0.5]],
                              [[3.], [3.], [3.]]])

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
                        [[3.0*3*(acc+vel*self.dt)], [3.0*vel*3], [3.0*3*(acc+vel*self.dt)], [3.0*vel*3], [3.0*3*(acc+vel*self.dt)], [3.0*vel*3]]])

        exp_x = np.array([[[0.], [0.], [0.], [0.], [0.], [0.]],
                          [[2.+self.dt*3], [1.], [5.], [0.], [-1.-2.*self.dt*3], [-2.]],
                          [[0.5+0.5*self.dt*3], [0.5], [0.5+0.5*self.dt*3], [0.5], [0.5+0.5*self.dt*3], [0.5]],
                          [[1.], [0.], [1.], [0.], [1.], [0.]],
                          [[-1.+0.5*self.dt*3], [0.5], [-3.+2.*self.dt*3], [2.], [0.], [0.]]])

        exp = B_u + exp_x

        pred = model.buildStepGraph("", state_in, action_in)
        pred = model.buildStepGraph("", pred, action_in)
        pred = model.buildStepGraph("", pred, action_in)

        self.assertAllClose(exp, pred)

    def test_training(self):
        s = 4
        a = 2
        m = 1.5
        model = PointMassModel(m, self.dt, s, a)

        x = np.array([[0.5], [0.1], [0.2], [-0.1]])
        u = np.array([[0.2], [-0.15]])
        gt = np.array([[0.514], [0.14], [0.187], [-0.13]])

        model.train_step(gt, x, u)

        mt = model.getMass()

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
        self.params["linear_damping_forward_speed"] = [0., 0., 0., 0., 0., 0.]
        self.inertial = dict()
        self.inertial["ixx"] = 650.0
        self.inertial["iyy"] = 750.0
        self.inertial["izz"] = 550.0
        self.inertial["ixy"] = 1.0
        self.inertial["ixz"] = 2.0
        self.inertial["iyz"] = 3.0
        self.params["inertial"] = self.inertial
        self.model_quat = AUVModel(quat=True, action_dim=6, dt=0.1, k=1, parameters=self.params)
        self.model_euler = AUVModel(quat=False, action_dim=6, dt=0.1, k=1, parameters=self.params)

    def test_B2I_transform(self):
        ''' 
            Quat representation is [w, x, y, z] 
            quat: shape [k, 13, 1]
            euler representation is [roll, pitch, yaw]
            euler: shape [k, 12, 1]
        '''

        quat = np.array([[[0.], [0.], [0.], [1.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]],
                                  [[0.], [0.], [0.], [0.950510320581509], [0.0438308910967523], [0.25508068761447], [0.171880267220619], [0.], [0.], [0.], [0.], [0.], [0.]],
                                  [[0.], [0.], [0.], [0.586824089619078], [-0.111618880991033], [0.633022223770408], [0.492403876367579], [0.], [0.], [0.], [0.], [0.], [0.]]])


        euler = np.array([[[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]],
                                   [[0.], [0.], [0.], [0.1949573], [0.4891167], [0.4065898], [0.], [0.], [0.], [0.], [0.], [0.]],
                                   [[0.], [0.], [0.], [1.2317591], [1.0214548], [2.1513002], [0.], [0.], [0.], [0.], [0.], [0.]]])

        inp = np.array([[[0.], [0.], [0.], [0.], [0.], [0.]]])
        pose_quat, speed_quat = self.model_quat.prepare_data(quat)
        pose_euler, speed_euler = self.model_euler.prepare_data(euler)

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
                                ])
        wid=3
        xid=4
        yid=5
        zid=6

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
                           ])


        exp_TB2Iquat = np.zeros(shape=(3, 4, 3))

        exp_TB2Iquat[:, 0, 0] = -quat[:, xid, 0]
        exp_TB2Iquat[:, 0, 1] = -quat[:, yid, 0]
        exp_TB2Iquat[:, 0, 2] = -quat[:, zid, 0]

        exp_TB2Iquat[:, 1, 0] = quat[:, wid, 0]
        exp_TB2Iquat[:, 1, 1] = -quat[:, zid, 0]
        exp_TB2Iquat[:, 1, 2] = quat[:, yid, 0]

        exp_TB2Iquat[:, 2, 0] = quat[:, zid, 0]
        exp_TB2Iquat[:, 2, 1] = quat[:, wid, 0]
        exp_TB2Iquat[:, 2, 2] = -quat[:, xid, 0]

        exp_TB2Iquat[:, 3, 0] = -quat[:, yid, 0]
        exp_TB2Iquat[:, 3, 1] = quat[:, xid, 0]
        exp_TB2Iquat[:, 3, 2] = quat[:, wid, 0]
        
        exp_TB2Iquat = 0.5*exp_TB2Iquat
        # Euler section 
        rid=3
        pid=4
        yid=5


        cr = np.cos(euler[:, rid, 0])
        cp = np.cos(euler[:, pid, 0])
        cy = np.cos(euler[:, yid, 0])

        sr = np.sin(euler[:, rid, 0])
        sp = np.sin(euler[:, pid, 0])
        sy = np.sin(euler[:, yid, 0])



        exp_rot_euler = np.array([
                                  [
                                   [cy[0]*cp[0], -sy[0]*cr[0]+cy[0]*sp[0]*sr[0], sy[0]*sr[0]+cy[0]*cr[0]*sp[0]],
                                   [sy[0]*cp[0], cy[0]*cr[0]+sr[0]*sp[0]*sy[0], -cy[0]*sr[0]+sp[0]*sy[0]*cr[0]],
                                   [-sp[0], cp[0]*sr[0], cp[0]*cr[0]]
                                  ],

                                  [
                                   [cy[1]*cp[1], -sy[1]*cr[1]+cy[1]*sp[1]*sr[1], sy[1]*sr[1]+cy[1]*cr[1]*sp[1]],
                                   [sy[1]*cp[1], cy[1]*cr[1]+sr[1]*sp[1]*sy[1], -cy[1]*sr[1]+sp[1]*sy[1]*cr[1]],
                                   [-sp[1], cp[1]*sr[1], cp[1]*cr[1]]
                                  ],

                                  [
                                   [cy[2]*cp[2], -sy[2]*cr[2]+cy[2]*sp[2]*sr[2], sy[2]*sr[2]+cy[2]*cr[2]*sp[2]],
                                   [sy[2]*cp[2], cy[2]*cr[2]+sr[2]*sp[2]*sy[2], -cy[2]*sr[2]+sp[2]*sy[2]*cr[2]],
                                   [-sp[2], cp[2]*sr[2], cp[2]*cr[2]]
                                  ]
                                 ])

        exp_TB2Ieuler = np.zeros(shape=(3, 3, 3))

        exp_TB2Ieuler[:, 0, 0] = 1.
        exp_TB2Ieuler[:, 0, 1] = sr*sp/cp
        exp_TB2Ieuler[:, 0, 2] = cr*sp/cp
        
        exp_TB2Ieuler[:, 1, 1] = cr
        exp_TB2Ieuler[:, 1, 2] = -sr
        
        exp_TB2Ieuler[:, 2, 1] = sr/cp
        exp_TB2Ieuler[:, 2, 2] = cr/cp

        self.model_quat.body2inertial_transform_q(pose_quat)
        self.model_euler.body2inertial_transform(pose_euler)

        self.assertAllClose(self.model_quat._rotBtoI, exp_rot)
        self.assertAllClose(self.model_quat._rotBtoI, exp_rot_euler)
        self.assertAllClose(self.model_quat._rotBtoI, rot_from_lib)

        self.assertAllClose(self.model_euler._rotBtoI, exp_rot)
        self.assertAllClose(self.model_euler._rotBtoI, exp_rot_euler)
        self.assertAllClose(self.model_euler._rotBtoI, rot_from_lib)
        
        self.assertAllClose(self.model_quat._TBtoIquat, exp_TB2Iquat)

        self.assertAllClose(self.model_euler._TBtoIeuler, exp_TB2Ieuler)

    def test_restoring(self):
        pose = np.array([[[1.0], [1.0], [1.0], [0.0], [0.0], [0.0]]])
        self.model_euler.body2inertial_transform(pose)
        rest = self.model_euler.restoring_forces("rest")
        #print("*"*10 + " Rest " + "*"*10)
        #print(rest)
        #print("*"*20)
        pass

    def test_damping(self):
        pass
        vel = np.array([[[1.0], [1.0], [1.0], [0.0], [0.0], [0.0]]])
        d = self.model_euler.damping_matrix("damp", vel)

        #print("*"*10 + " Damping " + "*"*10)
        #print(d)
        #print("*"*20)
        pass

    def test_corrolis(self):
        vel = np.array([[[1.0], [1.0], [1.0], [0.0], [0.0], [0.0]]])
        Iv = [self.inertial["ixx"]*vel[0, 3, 0] - self.inertial["ixy"]*vel[0, 4, 0] - self.inertial["ixz"]*vel[0, 5, 0], 
              - self.inertial["ixy"]*vel[0, 3, 0] + self.inertial["iyy"]*vel[0, 4, 0] - self.inertial["iyz"]*vel[0, 5, 0],
              - self.inertial["ixz"]*vel[0, 3, 0] - self.inertial["iyz"]*vel[0, 4, 0] + self.inertial["izz"]*vel[0, 5, 0]]
        Mav = - np.dot(np.array(self.params["Ma"]), vel)

        crb = np.array([[[0., 0., 0., 0., self.params["mass"]*vel[0, 2, 0], -self.params["mass"]*vel[0, 1, 0]],
                        [0., 0., 0., -self.params["mass"]*vel[0, 2, 0], 0., self.params["mass"]*vel[0, 0, 0]],
                        [0., 0., 0., self.params["mass"]*vel[0, 1, 0], -self.params["mass"]*vel[0, 0, 0], 0.],
                        [0., self.params["mass"]*vel[0, 2, 0], -self.params["mass"]*vel[0, 1, 0], 0., Iv[2], -Iv[1]],
                        [-self.params["mass"]*vel[0, 2, 0], 0., self.params["mass"]*vel[0, 0, 0], -Iv[2], 0., Iv[0]],
                        [self.params["mass"]*vel[0, 1, 0], -self.params["mass"]*vel[0, 0, 0], 0., Iv[1], -Iv[0], 0.]]])

        ca = np.array([[[0., 0., 0., 0., -Mav[2, 0, 0], Mav[1, 0, 0]],
                       [0., 0., 0., Mav[2, 0, 0], 0., -Mav[0, 0, 0]],
                       [0., 0., 0., -Mav[1, 0, 0], Mav[0, 0, 0], 0.],
                       [0., -Mav[2, 0, 0], Mav[1, 0, 0], 0., -Mav[5, 0, 0], Mav[4, 0, 0]],
                       [Mav[2, 0, 0], 0., -Mav[0, 0, 0], Mav[5, 0, 0], 0., -Mav[3, 0, 0]],
                       [-Mav[1, 0, 0], Mav[0, 0, 0], 0., -Mav[4, 0, 0], Mav[3, 0, 0], 0.]]])

        c = self.model_euler.coriolis_matrix("coriolis", vel)

        self.assertAllClose(c, crb + ca)
        pass

    def test_step1_k1_s12_a6(self):
        state = np.array([[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                           [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]])
        action = np.array([[[1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]])
        next_state = self.model_euler.buildStepGraph("step", state, action)

        #print("*"*10 + " Next State " + "*"*10)
        #print(next_state)
        #print("*"*20)
        pass

    def test_step1_k5_s12_a6(self):
        state = np.array([ [ [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                             [0.0], [0.0], [0.0], [0.0], [0.0], [0.0] ],

                           [ [1.0], [1.0], [1.0], [0.0], [0.0], [0.0],
                             [0.1], [2.0], [2.0], [1.0], [2.0], [3.0] ],

                           [ [0.0], [2.0], [1.0], [0.2], [0.3], [0.0],
                             [-1.], [-1.], [-1.], [-1.], [-1.], [-1.] ],

                           [ [5.0], [0.2], [0.0], [1.2], [0.0], [3.1],
                             [0.0], [0.0], [0.0], [0.0], [0.0], [0.0] ],

                           [ [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                             [1.0], [1.0], [1.0], [1.0], [1.0], [1.0] ] ])

        action = np.array([[ [1.0], [1.0], [1.0], [1.0], [1.0], [1.0] ],
                           [ [1.0], [1.0], [1.0], [1.0], [1.0], [1.0] ],
                           [ [-1.], [-1.], [-1.], [-1.], [-1.], [-1.] ],
                           [ [2.0], [2.0], [2.0], [2.0], [2.0], [2.0] ],
                           [ [-1.], [-1.], [-1.], [-1.], [-1.], [-1.] ]])

        next_state = self.model_euler.buildStepGraph("step", state, action)
        #print("*"*10 + " Next State Many " + "*"*10)
        #print(next_state)
        #print("*"*20)
        pass

class TestCost(tf.test.TestCase):
    def setUp(self):
        self.dt=0.1

    def testStepCost_s2_a2_l1(self):

        state=np.array([[[0.], [1.]]])
        action=np.array([[1.], [1.]])
        noise=np.array([[[1.], [1.]]])
        Sigma=np.array([[1., 0.], [0., 1.]])
        lam=np.array([1.])
        gamma = 1.
        upsilon = 1.

        cost = CostBase(lam, gamma, upsilon, Sigma)

        exp_a_c = np.array([[[ 0.5*(gamma*(2. + 4.) + lam[0]*(1-1./upsilon)*(0.) ) ]]])


        with self.assertRaises(NotImplementedError):
            _ = cost.state_cost("", state)

        a_c_dict = cost.action_cost("", action, noise)
        self.assertAllClose(exp_a_c, a_c_dict["action_cost"])

    def testStepCost_s4_a2_l1(self):

        state=np.array([[[0.], [0.5], [2.], [0.]]])
        action=np.array([[0.5], [2.]])
        noise=np.array([[[0.5], [1.]]])
        Sigma=np.array([[1., 0.], [0., 1.]])

        lam=np.array([1.])
        lam=np.array([1.])
        gamma = 1.
        upsilon = 1.

        cost = CostBase(lam, gamma, upsilon, Sigma)

        exp_a_c = np.array([[[0.5*(gamma*(4.25 + 2.*2.25) + lam[0]*(1-1./upsilon)*(1.25) )]]])

        with self.assertRaises(NotImplementedError):
            _ = cost.state_cost("", state)

        a_c_dict = cost.action_cost("", action, noise)
        self.assertAllClose(exp_a_c, a_c_dict["action_cost"])

    def testStepCost_s4_a3_l1(self):

        state=np.array([[[0.], [0.5], [2.], [0.]],
                        [[0.], [2.], [0.], [0.]],
                        [[10.], [2.], [2.], [3.]],
                        [[1.], [1.], [1.], [2.]],
                        [[3.], [4.], [5.], [6.]]])
        action=np.array([[0.5], [2.], [0.25]])
        noise=np.array([[[0.5], [1.], [2.]],
                        [[0.5], [2.], [0.25]],
                        [[-2], [-0.2], [-1]],
                        [[0.], [0.], [0.]],
                        [[1.], [0.5], [3.]]])
        Sigma=np.array([[1., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 1.]])
        lam=np.array([1.])
        gamma = 1.
        upsilon = 1.

        cost = CostBase(lam, gamma, upsilon, Sigma)


        exp_a_c = np.array([[[0.5*(gamma*(4.3125 + 2.*2.75) + lam[0]*(1-1./upsilon)*(5.25) ) ]],
                            [[0.5*(gamma*(4.3125 + 2.*4.3125) + lam[0]*(1-1./upsilon)*(4.3125) ) ]],
                            [[0.5*(gamma*(4.3125 + 2.*-1.65) + lam[0]*(1-1./upsilon)*(5.04) ) ]],
                            [[0.5*(gamma*(4.3125 + 2.*0) + lam[0]*(1-1./upsilon)*(0) ) ]],
                            [[0.5*(gamma*(4.3125 + 2.*2.25) + lam[0]*(1-1./upsilon)*(10.25) ) ]]])


        with self.assertRaises(NotImplementedError):
            _ = cost.state_cost("", state)

        a_c_dict = cost.action_cost("", action, noise)
        self.assertAllClose(exp_a_c, a_c_dict["action_cost"])

    def testStepCost_s4_a3_l10_g2_u3(self):

        state=np.array([[[0.], [0.5], [2.], [0.]],
                        [[0.], [2.], [0.], [0.]],
                        [[10.], [2.], [2.], [3.]],
                        [[1.], [1.], [1.], [2.]],
                        [[3.], [4.], [5.], [6.]]])
        action=np.array([[0.5], [2.], [0.25]])
        noise=np.array([[[0.5], [1.], [2.]],
                        [[0.5], [2.], [0.25]],
                        [[-2], [-0.2], [-1]],
                        [[0.], [0.], [0.]],
                        [[1.], [0.5], [3.]]])
        Sigma=np.array([[1., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 1.]])
        lam=np.array([10.])
        gamma = 2.
        upsilon = 3.

        cost = CostBase(lam, gamma, upsilon, Sigma)


        exp_a_c = np.array([[[0.5*(gamma*(4.3125 + 2.*2.75) + lam[0]*(1-1./upsilon)*(5.25) ) ]],
                            [[0.5*(gamma*(4.3125 + 2.*4.3125) + lam[0]*(1-1./upsilon)*(4.3125) ) ]],
                            [[0.5*(gamma*(4.3125 + 2.*-1.65) + lam[0]*(1-1./upsilon)*(5.04) ) ]],
                            [[0.5*(gamma*(4.3125 + 2.*0) + lam[0]*(1-1./upsilon)*(0) ) ]],
                            [[0.5*(gamma*(4.3125 + 2.*2.25) + lam[0]*(1-1./upsilon)*(10.25) ) ]]])


        with self.assertRaises(NotImplementedError):
            _ = cost.state_cost("", state)

        a_c_dict = cost.action_cost("", action, noise)
        self.assertAllClose(exp_a_c, a_c_dict["action_cost"])

    def testStepCost_s4_a3_l15_g20_u30(self):

        state=np.array([[[0.], [0.5], [2.], [0.]],
                        [[0.], [2.], [0.], [0.]],
                        [[10.], [2.], [2.], [3.]],
                        [[1.], [1.], [1.], [2.]],
                        [[3.], [4.], [5.], [6.]]])
        action=np.array([[0.5], [2.], [0.25]])
        noise=np.array([[[0.5], [1.], [2.]],
                        [[0.5], [2.], [0.25]],
                        [[-2], [-0.2], [-1]],
                        [[0.], [0.], [0.]],
                        [[1.], [0.5], [3.]]])
        Sigma=np.array([[1., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 1.]])
        lam=np.array([15.])
        gamma = 20.
        upsilon = 30.

        cost = CostBase(lam, gamma, upsilon, Sigma)


        exp_a_c = np.array([[[0.5*(gamma*(4.3125 + 2.*2.75) + lam[0]*(1-1./upsilon)*(5.25) ) ]],
                            [[0.5*(gamma*(4.3125 + 2.*4.3125) + lam[0]*(1-1./upsilon)*(4.3125) ) ]],
                            [[0.5*(gamma*(4.3125 + 2.*-1.65) + lam[0]*(1-1./upsilon)*(5.04) ) ]],
                            [[0.5*(gamma*(4.3125 + 2.*0) + lam[0]*(1-1./upsilon)*(0) ) ]],
                            [[0.5*(gamma*(4.3125 + 2.*2.25) + lam[0]*(1-1./upsilon)*(10.25) ) ]]])


        with self.assertRaises(NotImplementedError):
            _ = cost.state_cost("", state)

        a_c_dict = cost.action_cost("", action, noise)
        self.assertAllClose(exp_a_c, a_c_dict["action_cost"])


class TestStaticCost(tf.test.TestCase):
    def setUp(self):
        self.dt=0.1

    def testStepStaticCost_s2_a2_l1(self):

        state=np.array([[[0.], [1.]]])
        goal=np.array([[1.], [1.]])
        action=np.array([[1.], [1.]])
        noise=np.array([[[1.], [1.]]])
        Sigma=np.array([[1., 0.], [0., 1.]])
        Q=np.array([[1., 0.], [0., 1.]])
        lam=np.array([1.])
        gamma = 1.
        upsilon = 1.

        cost = StaticCost(lam, gamma, upsilon, Sigma, goal, Q)

        exp_a_c = np.array([[[ 0.5*(gamma*(2. + 4.) + lam[0]*(1-upsilon)*(0.) ) ]]])
        exp_s_c = np.array([[[1.]]])


        exp_c = exp_a_c + exp_s_c

        a_c_dict = cost.action_cost("", action, noise)
        c_dict = cost.build_step_cost_graph("", state, action, noise)

        self.assertAllClose(exp_a_c, a_c_dict["action_cost"])
        self.assertAllClose(exp_s_c, c_dict["state_cost"])
        self.assertAllClose(exp_c, c_dict["cost"])

    def testStepStaticCost_s4_a2_l1(self):

        state=np.array([[[0.], [0.5], [2.], [0.]]])
        goal=np.array([[1.], [1.], [1.], [2.]])
        action=np.array([[0.5], [2.]])
        noise=np.array([[[0.5], [1.]]])
        Sigma=np.array([[1., 0.], [0., 1.]])
        Q=np.array([[1., 0., 0., 0.],
                   [0., 1., 0., 0.],
                   [0., 0., 10., 0.],
                   [0., 0., 0., 10.]])

        lam=np.array([1.])
        gamma = 1.
        upsilon = 1.

        cost = StaticCost(lam, gamma, upsilon, Sigma, goal, Q)

        exp_a_c = np.array([[[0.5*(gamma*(4.25 + 2.*2.25) + lam[0]*(1-upsilon)*(1.25) )]]])

        exp_s_c = np.array([[[51.25]]])

        exp_c = exp_a_c + exp_s_c


        a_c_dict = cost.action_cost("", action, noise)
        c_dict = cost.build_step_cost_graph("", state, action, noise)

        self.assertAllClose(exp_a_c, a_c_dict["action_cost"])
        self.assertAllClose(exp_s_c, c_dict["state_cost"])
        self.assertAllClose(exp_c, c_dict["cost"])

    def testStepStaticCost_s4_a3_l1(self):

        state=np.array([[[0.], [0.5], [2.], [0.]],
                        [[0.], [2.], [0.], [0.]],
                        [[10.], [2.], [2.], [3.]],
                        [[1.], [1.], [1.], [2.]],
                        [[3.], [4.], [5.], [6.]]])
        goal=np.array([[1.], [1.], [1.], [2.]])
        action=np.array([[0.5], [2.], [0.25]])
        noise=np.array([[[0.5], [1.], [2.]],
                        [[0.5], [2.], [0.25]],
                        [[-2], [-0.2], [-1]],
                        [[0.], [0.], [0.]],
                        [[1.], [0.5], [3.]]])
        Sigma=np.array([[1., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 1.]])
        Q=np.array([[1., 0., 0., 0.],
                   [0., 1., 0., 0.],
                   [0., 0., 10., 0.],
                   [0., 0., 0., 10.]])
        lam=np.array([1.])
        gamma = 1.
        upsilon = 1.

        cost = StaticCost(lam, gamma, upsilon, Sigma, goal, Q)

        exp_a_c = np.array([[[0.5*(gamma*(4.3125 + 2.*2.75) + lam[0]*(1-upsilon)*(5.25) ) ]],
                            [[0.5*(gamma*(4.3125 + 2.*4.3125) + lam[0]*(1-upsilon)*(4.3125) ) ]],
                            [[0.5*(gamma*(4.3125 + 2.*-1.65) + lam[0]*(1-upsilon)*(5.04) ) ]],
                            [[0.5*(gamma*(4.3125 + 2.*0) + lam[0]*(1-upsilon)*(0) ) ]],
                            [[0.5*(gamma*(4.3125 + 2.*2.25) + lam[0]*(1-upsilon)*(10.25) ) ]]])

        exp_s_c = np.array([[[51.25]], [[52]], [[102]], [[0.]], [[333]]])

        exp_c = exp_a_c + exp_s_c

        a_c_dict = cost.action_cost("", action, noise)
        c_dict = cost.build_step_cost_graph("", state, action, noise)

        self.assertAllClose(exp_a_c, a_c_dict["action_cost"])
        self.assertAllClose(exp_s_c, c_dict["state_cost"])
        self.assertAllClose(exp_c, c_dict["cost"])


class TestElipseCost(tf.test.TestCase):
    def setUp(self):
        self.dt=0.1
        self.tau=1
        self.a=1
        self.b=1
        self.cx=0
        self.cy=0
        self.r=1
        self.s=1
        self.m_s=1
        self.m_v=1

    def testStepElipseCost_s4_l1_k1(self):

        state=np.array([[[0.], [0.5], [1.], [0.]]])
        sigma=np.array([[1., 0.], [0., 1.]])
        lam=np.array([1.])
        gamma = 1.
        upsilon = 1.

        cost = ElipseCost(lam,
                          gamma,
                          upsilon,
                          sigma,
                          self.a, self.b, self.cx, self.cy,
                          self.s, self.m_s, self.m_v)

        exp_s_c = np.array([[[0.25]]])

        s_c_dict = cost.state_cost("", state)


        self.assertAllClose(exp_s_c, s_c_dict["state_cost"])

    def testStepElipseCost_s4_l1_k5(self):

        state=np.array([[[0.], [0.5], [1.], [0.]],
                        [[0.], [2.], [0.], [0.]],
                        [[10.], [2.], [2.], [3.]],
                        [[1.], [1.], [1.], [2.]],
                        [[3.], [4.], [5.], [6.]]])
        sigma=np.array([[1., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 1.]])
        lam=np.array([1.])
        gamma = 1.
        upsilon = 1.

        cost = ElipseCost(lam,
                          gamma,
                          upsilon,
                          sigma,
                          self.a, self.b, self.cx, self.cy,
                          self.s, self.m_s, self.m_v)

        exp_s_c = np.array([[[0.25]],
                            [[2]],
                            [[103+6.788897449072021]],
                            [[1+1.5278640450004208]],
                            [[33+38.57779489814404]]])

        s_c_dict = cost.state_cost("", state)
        self.assertAllClose(exp_s_c, s_c_dict["state_cost"])


class TestController(tf.test.TestCase):
    def setUp(self):
        self.k = 5
        self.tau = 3
        self.a_dim = 2
        self.s_dim = 4
        self.dt = 0.01
        self.mass = 1.
        self.lam = 1.
        self.gamma = 1.
        self.upsilon = 1.

        self.goal = np.array([[0.], [1.], [0.], [1.]])
        self.sigma = np.array([[1., 0.], [0., 1.]])
        self.Q = np.array([[1., 0., 0., 0.],
                           [0., 1., 0., 0.],
                           [0., 0., 1., 0.],
                           [0., 0., 0., 1.]])

        self.c = np.array([[[3.]], [[10.]], [[0.]], [[1.]], [[5.]]])
        self.n = np.array([[[[1.], [-0.5]], [[1.], [-0.5]], [[2.], [1.]]],
                           [[[0.3], [0.]], [[2.], [0.2]], [[1.2], [3.]]],
                           [[[0.5], [0.5]], [[0.5], [0.5]], [[0.5], [0.5]]],
                           [[[0.6], [0.7]], [[0.2], [-0.3]], [[0.1], [-0.4]]],
                           [[[-2.], [-3.]], [[-4.], [-1.]], [[0.], [0.]]]])

        self.a = np.array([[[1.], [0.5]], [[2.3], [4.5]], [[2.1], [-0.4]]])
        model = PointMassModel(self.mass, self.dt, self.s_dim, self.a_dim)
        cost = StaticCost(self.lam, self.gamma, self.upsilon, self.sigma, self.goal, self.Q)
        self.cont = ControllerBase(model,
                                   cost,
                                   self.k,
                                   self.tau,
                                   self.dt,
                                   self.s_dim,
                                   self.a_dim,
                                   self.lam,
                                   self.sigma)

    def testDataPrep(self):
        exp_a0 = np.array([[1.], [0.5]])
        exp_a1 = np.array([[2.3], [4.5]])
        exp_a2 = np.array([[2.1], [-0.4]])
        exp_n0 = np.array([[[1.], [-0.5]], [[0.3], [0.]], [[0.5], [0.5]], [[0.6], [0.7]], [[-2.], [-3.]]])
        exp_n1 = np.array([[[1.], [-0.5]], [[2.], [0.2]], [[0.5], [0.5]], [[0.2], [-0.3]], [[-4], [-1]]])
        exp_n2 = np.array([[[2.], [1.]], [[1.2], [3.]], [[0.5], [0.5]], [[0.1], [-0.4]], [[0.], [0.]]])


        a0 = self.cont.prepareAction("", self.a, 0)
        n0 = self.cont.prepareNoise("", self.n, 0)

        a1 = self.cont.prepareAction("", self.a, 1)
        n1 = self.cont.prepareNoise("", self.n, 1)

        a2 = self.cont.prepareAction("", self.a, 2)
        n2 = self.cont.prepareNoise("", self.n, 2)

        self.assertAllClose(a0, exp_a0)
        self.assertAllClose(n0, exp_n0)

        self.assertAllClose(a1, exp_a1)
        self.assertAllClose(n1, exp_n1)

        self.assertAllClose(a2, exp_a2)
        self.assertAllClose(n2, exp_n2)

    def testUpdate(self):
        beta = np.array([[0]])
        exp_arg = np.array([[[-3.]], [[-10.]], [[0.]], [[-1.]], [[-5.]]])
        exp = np.array([[[0.049787068367863944]],
                        [[4.5399929762484854e-05]],
                        [[1]],
                        [[0.36787944117144233]],
                        [[0.006737946999085467]]])

        nabla = np.array([[1.424449856468154]])
        weights = np.array([[[0.034951787275480706]],
                           [[3.1871904480408675e-05]],
                           [[0.7020254138530686]],
                           [[0.2582607169364174]],
                           [[0.004730210030553017]]])

        expected = np.array([[[0.034951787275480706*1. + 3.1871904480408675e-05*0.3 + 0.7020254138530686*0.5 + 0.2582607169364174*0.6 + 0.004730210030553017*(-2)],
                              [0.034951787275480706*(-0.5) + 3.1871904480408675e-05*0 + 0.7020254138530686*0.5 + 0.2582607169364174*0.7 + 0.004730210030553017*(-3)]],
                             [[0.034951787275480706*1 + 3.1871904480408675e-05*2 + 0.7020254138530686*0.5 + 0.2582607169364174*0.2 + 0.004730210030553017*(-4)],
                              [0.034951787275480706*(-0.5) + 3.1871904480408675e-05*0.2 + 0.7020254138530686*0.5 + 0.2582607169364174*(-0.3) + 0.004730210030553017*(-1)]],
                             [[0.034951787275480706*2 + 3.1871904480408675e-05*1.2 + 0.7020254138530686*0.5 + 0.2582607169364174*0.1 + 0.004730210030553017*0],
                              [0.034951787275480706*1 + 3.1871904480408675e-05*3 + 0.7020254138530686*0.5 + 0.2582607169364174*(-0.4) + 0.004730210030553017*0]]])

        b = self.cont.beta("", self.c)
        arg = self.cont.normArg("", self.c, b, False)
        e_arg = self.cont.expArg("", arg)
        e = self.cont.exp("", e_arg)
        nab = self.cont.nabla("", e)
        w = self.cont.weights("", e, nab)
        w_n = self.cont.weightedNoise("", w, self.n)
        sum = tf.reduce_sum(w)


        self.assertAllClose(b, beta)
        self.assertAllClose(e_arg, exp_arg)

        self.assertAllClose(e, exp)
        self.assertAllClose(nab, nabla)

        self.assertAllClose(w, weights)
        self.assertAllClose(w_n, expected)
        self.assertAllClose(sum, 1.)

    def testNew(self):
        next1 = np.array([[[1.], [0.5]]])
        next2 = np.array([[[1.], [0.5]], [[2.3], [4.5]]])
        next3 = np.array([[[1.], [0.5]], [[2.3], [4.5]], [[2.1], [-0.4]]])

        n1 = self.cont.getNext("", self.a, 1)
        n2 = self.cont.getNext("", self.a, 2)
        n3 = self.cont.getNext("", self.a, 3)

        self.assertAllClose(n1, next1)
        self.assertAllClose(n2, next2)
        self.assertAllClose(n3, next3)

    def testShiftAndInit(self):
        init1 = np.array([[[1.], [0.5]]])
        init2 = np.array([[[1.], [0.5]], [[2.3], [4.5]]])

        exp1 = np.array([[[2.3], [4.5]], [[2.1], [-0.4]], [[1.], [0.5]]])
        exp2 = np.array([[[2.1], [-0.4]], [[1.], [0.5]], [[2.3], [4.5]]])

        n1 = self.cont.shift("", self.a, init1, 1)
        n2 = self.cont.shift("", self.a, init2, 2)

        self.assertAllClose(n1, exp1)
        self.assertAllClose(n2, exp2)

    def testAll(self):
        pass


tf.test.main()
