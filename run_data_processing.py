import os
from os import listdir, mkdir
from os.path import isfile, join, exists
import numpy as np
import pandas as pd
import shutil
import yaml
from tqdm import tqdm
from scripts.utils.utils import parse_param

import argparse
from bagpy import bagreader
from scipy.spatial.transform import Rotation as R


old_state_labels = ['pose.pose.position.x',
                   'pose.pose.position.y',
                   'pose.pose.position.z',
                   'pose.pose.orientation.x',
                   'pose.pose.orientation.y',
                   'pose.pose.orientation.z',
                   'pose.pose.orientation.w',
                   'twist.twist.linear.x',
                   'twist.twist.linear.y',
                   'twist.twist.linear.z',
                   'twist.twist.angular.x',
                   'twist.twist.angular.y',
                   'twist.twist.angular.z']
new_state_labels = ["x", "y", "z", "qx", "qy", "qz", "qw", "Su", "Sv", "Sw", "Sp", "Sq", "Sr"]

old_action_labels = ["wrench.force.x",
                     "wrench.force.y",
                     "wrench.force.z",
                     "wrench.torque.x",
                     "wrench.torque.y",
                     "wrench.torque.z",]
new_action_labels = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]


'''
    Main function. Cleans a set of bags contained in directory dataDir.
    The bag is cleaned, resized and resampled. If the bag is corrupted
    it is moved to a directory, within the dataDir, named corrupted.

    inputs:
    -------
        - dataDir string, the directory containing the bags to clean.
        - outDir string, saving directory for the csv files.
        - steps int, the number of transition to keep. The size 
            of the bag is min(n, len(bag))
        - freq float, the time frequency of the resampled 
            bag. Expressed in hertz.
        - norm string, yaml file containing vehicle information.
'''
def clean_bags(data_dir, out_dir, steps=500, frequency=10, norm=None):
    # load bag.
    corruptDir = join(data_dir, 'corrupted')
    if not exists(corruptDir):
        mkdir(corruptDir)
    if not exists(out_dir):
        os.makedirs(out_dir)

    files = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
    t = tqdm(files, desc="Cleaning", ncols=150, colour="green", postfix={"corrupted:": None})

    for f in t:
        bag_file = join(data_dir, f)
        name = os.path.splitext(f)[0]
        bag_dir = join(data_dir, name)
        corrupt = join(data_dir, "corrupted", f)
        if exists(bag_file):
            state_data, action_data = dataframe_from_bag(bag_file)
            # Resample can fail if there is some issue with the bag.
            try:
                state_data = resample(state_data, old_state_labels, frequency)
                action_data = resample(action_data, old_action_labels, frequency)
            except:
                t.set_postfix({"corrupted:": f"{f}"})
                os.rename(bag_file, corrupt)
                if exists(bag_dir):
                    shutil.rmtree(bag_dir)
                continue
            
            state_data = relabel(state_data, old_state_labels, new_state_labels)
            action_data = relabel(action_data, old_action_labels, new_action_labels)

            state_data = compute_angles_representation(state_data)
            state_data = compute_body_vels(state_data)
            state_data = compute_inertial_vels(state_data)
            state_data = compute_delta_vels(state_data)

            action_data = compute_norm_actions(action_data, norm)

            cleaned_data = pd.concat([state_data, action_data], axis=1)
            cleaned_data = cleaned_data.set_index(np.arange(len(cleaned_data)))
            if (steps is not None) and steps < len(cleaned_data):
                cleaned_data = cleaned_data[:steps]
            columns = cleaned_data.columns
        
            pd.DataFrame(data=cleaned_data, columns=columns).to_csv(os.path.join(out_dir, name + ".csv"))

'''
    Extracts dataframe from bag.

    inputs:
    -------
        - path str, the absolute/relative path to the rosbag.
        - model_name, name of the AUV to pick the right topics.
'''
def dataframe_from_bag(path, model_name="rexrov2"):
    bag = bagreader(path, verbose=False)
    df_state = pd.read_csv(bag.message_by_topic(f"/{model_name}/pose_gt"))
    df_action = pd.read_csv(bag.message_by_topic(f"/thruster_input"))
    return df_state, df_action

'''
    Resamples the rostopics of the dataframe at the desired frequency.

    inputs:
    -------
        - dataframe: pandas.dataframe. The dataframe containing the desired topics.
        - topics_name: list, the interesting topic names.
        - frequency: float, the desired frequency of the topics.
'''
def resample(dataframe, topics_name, frequency):
    period = 1./frequency
    entries = topics_name.copy()
    entries.append("Time")
    df = dataframe.loc[:, entries]
    # relative time of a traj as all the trajs are captured in the same gazebo instance.
    df['Time'] = df['Time'] - df['Time'][0]
    df['Time'] = pd.to_datetime(df['Time'], unit='s').round('ms')
    traj = df.copy()
    traj.index = df['Time']
    traj.drop('Time', axis=1, inplace=True)
    traj = traj.resample('ms').interpolate('linear').resample(f'{period}S').interpolate()
    return traj

'''
    Renames the labels used in a dataframe.

    inputs:
    -------
        - dataframe: pandas.dataframe.
        - old_label: list. the labels that needs to be changed
        - new_label: list. The new labels. Needs to be the same lenght as old_lable
'''
def relabel(dataframe, old_label, new_label):
    mapping = {old_label[i]: new_label[i] for i in range(len(old_label))}
    dataframe.rename(columns=mapping, inplace=True)
    return dataframe

'''
    Computes multiple angles representation and adds them to the dataframe.

    inputs:
    -------
        - dataframe, pandas.dataframe. The dataframe with at least the entries
            "qx, qy, qz, qw" that represent a quaternion.
'''
def compute_angles_representation(dataframe):
    quats = dataframe.loc[:, ["qx", "qy", "qz", "qw"]].to_numpy()
    r = R.from_quat(quats)
    euler = r.as_euler("xyz", False)
    mat = r.as_matrix()
    rot_vec = r.as_rotvec()
    
    # Euler angles (radians)
    dataframe['roll'] = euler[:, 0]
    dataframe['pitch'] = euler[:, 1]
    dataframe['yaw'] = euler[:, 2]

    # Rotation Matrix
    dataframe['r00'] = mat[:, 0, 0]
    dataframe['r01'] = mat[:, 0, 1]
    dataframe['r02'] = mat[:, 0, 2]

    dataframe['r10'] = mat[:, 1, 0]
    dataframe['r11'] = mat[:, 1, 1]
    dataframe['r12'] = mat[:, 1, 2]

    dataframe['r20'] = mat[:, 2, 0]
    dataframe['r21'] = mat[:, 2, 1]
    dataframe['r22'] = mat[:, 2, 2]

    # Rotation Vector
    dataframe['rv0'] = rot_vec[:, 0]
    dataframe['rv1'] = rot_vec[:, 1]
    dataframe['rv2'] = rot_vec[:, 2]
    return dataframe


def compute_body_vels(dataframe):
    rotBtoI = dataframe.loc[:, ['r00', 'r01', 'r02', 'r10', 'r11', 'r12', 'r20', 'r21', 'r22']].to_numpy().reshape(-1, 3, 3)
    rotItoB = np.transpose(rotBtoI, axes=(0, 2, 1))
    sim_lin_vel = dataframe.loc[:, ['Su', 'Sv', 'Sw']].to_numpy()
    sim_ang_vel = dataframe.loc[:, ['Sp', 'Sq', 'Sr']].to_numpy()

    body_lin_vel = np.matmul(rotItoB, sim_lin_vel[..., None])[..., 0]
    body_ang_vel = np.matmul(rotItoB, sim_ang_vel[..., None])[..., 0]

    body_vel = np.concatenate([body_lin_vel, body_ang_vel], axis=-1)
    dataframe['Bu'] = body_vel[:, 0]
    dataframe['Bv'] = body_vel[:, 1]
    dataframe['Bw'] = body_vel[:, 2]
    dataframe['Bp'] = body_vel[:, 3]
    dataframe['Bq'] = body_vel[:, 4]
    dataframe['Br'] = body_vel[:, 5]
    return dataframe


def compute_inertial_vels(dataframe):
    sim_lin_vel = dataframe.loc[:, ['Su', 'Sv', 'Sw']].to_numpy()
    sim_ang_vel = dataframe.loc[:, ['Sp', 'Sq', 'Sr']].to_numpy()
    inertial_ang_vel = sim_ang_vel
    # translation
    t = dataframe.loc[:, ['x', 'y', 'z']].to_numpy()

    skewT = np.zeros((t.shape[0], 3, 3))

    skewT[:, 0, 1] = - t[:, 2]
    skewT[:, 1, 0] = t[:, 2]

    skewT[:, 0, 2] = t[:, 1]
    skewT[:, 2, 0] = - t[:, 1]

    skewT[:, 1, 2] = - t[:, 0]
    skewT[:, 2, 1] = t[:, 0]

    inertial_lin_vel = sim_lin_vel + np.matmul(skewT, sim_ang_vel[..., None])[..., 0]
    inertial_vel = np.concatenate([inertial_lin_vel, inertial_ang_vel], axis=-1)
    dataframe['Iu'] = inertial_vel[:, 0]
    dataframe['Iv'] = inertial_vel[:, 1]
    dataframe['Iw'] = inertial_vel[:, 2]
    dataframe['Ip'] = inertial_vel[:, 3]
    dataframe['Iq'] = inertial_vel[:, 4]
    dataframe['Ir'] = inertial_vel[:, 5]
    return dataframe


def compute_delta_vels(dataframe):
    inertial_vel = dataframe.loc[:, ['Iu', 'Iv', 'Iw', 'Ip', 'Iq', 'Ir']].to_numpy()
    inertial_dv = np.zeros(shape=(inertial_vel.shape))
    inertial_dv[1:] = inertial_vel[1:] - inertial_vel[:-1]

    dataframe['Idu'] = inertial_dv[:, 0]
    dataframe['Idv'] = inertial_dv[:, 1]
    dataframe['Idw'] = inertial_dv[:, 2]
    dataframe['Idp'] = inertial_dv[:, 3]
    dataframe['Idq'] = inertial_dv[:, 4]
    dataframe['Idr'] = inertial_dv[:, 5]


    body_vel = dataframe.loc[:, ['Bu', 'Bv', 'Bw', 'Bp', 'Bq', 'Br']].to_numpy()
    body_dv = np.zeros(shape=(body_vel.shape))
    body_dv[1:] = body_vel[1:] - body_vel[:-1]

    dataframe['Bdu'] = body_dv[:, 0]
    dataframe['Bdv'] = body_dv[:, 1]
    dataframe['Bdw'] = body_dv[:, 2]
    dataframe['Bdp'] = body_dv[:, 3]
    dataframe['Bdq'] = body_dv[:, 4]
    dataframe['Bdr'] = body_dv[:, 5]

    return dataframe


def compute_norm_actions(dataframe, norms):
    # Linear
    dataframe['Ux'] = norm_action(dataframe['Fx'], norms['x'])
    dataframe['Uy'] = norm_action(dataframe['Fy'], norms['y'])
    dataframe['Uz'] = norm_action(dataframe['Fz'], norms['z'])
    # Angular
    dataframe['Vx'] = norm_action(dataframe['Tx'], norms['p'])
    dataframe['Vy'] = norm_action(dataframe['Ty'], norms['q'])
    dataframe['Vz'] = norm_action(dataframe['Tz'], norms['r'])
    return dataframe


def norm_action(act, norm):
    # norm[0] is min
    # norm[1] is max
    # Norm between -1 and 1
    return (act - norm[0]) / (norm[1] - norm[0])*2 - 1


def compute_dv_stats(data_dir):
    files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    length = 0
    cummul = np.zeros(shape=(12))
    labels = ["Bdu", "Bdv", "Bdw", "Bdp", "Bdq", "Bdr",
              "Idu", "Idv", "Idw", "Idp", "Idq", "Idr"]
    for f in files:
        df = pd.read_csv(os.path.join(data_dir, f))
        dv = df.loc[:, labels].to_numpy()
        cummul += dv.sum(axis=0)
        length += len(dv)
    mean = cummul/length

    cummul = np.zeros(shape=(12))
    for f in files:
        df = pd.read_csv(os.path.join(data_dir, f))
        dv = df.loc[:, labels].to_numpy()
        cummul += np.power(dv-mean, 2).sum(axis=0)

    std = np.power(cummul/length, 0.5)
    stats = {"mean":
                {"B_norm": mean[:6].tolist(),
                 "I_norm": mean[6:].tolist()},
             "std":
                {"B_norm": std[:6].tolist(),
                 "I_norm": std[6:].tolist()}
            }

    save_dir = os.path.join(data_dir, "stats")
    if not exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, "stats.yaml"), 'w+') as stream:
        yaml.dump(stats, stream)


def parse_arg():
    parser = argparse.ArgumentParser(prog="clean_bag",
                                     description="Clean and resamples a set of rosbags\
                                        and saves them into a set of csv files.")

    parser.add_argument("-d", "--datadir", type=str, default=None,
                        help="Directory storing the rosbags.")

    parser.add_argument("-o", "--outdir", type=str, default=".",
                        help="Saving directory.")

    parser.add_argument("-f", "--frequency", type=float,
                        help="The desired frequency of the transitions withing the\
                            csv files. Default 10hz, tau=0.1s",
                        default=10)

    parser.add_argument("-s", "--steps", type=int,
                        help="number of transition steps in the csv file",
                        default=500)

    parser.add_argument("-n", "--norm", type=str,
                        help="Yaml file containing max and min thrust of the vehicle",
                        default=None)
    
    args = parser.parse_args()
    return args


def main():
    # parse arguments.
    args = parse_arg()
    if args.datadir is None:
        print("No datadir provided, nothing to clean.")
        return
    norm = parse_param(args.norm)
    clean_bags(args.datadir, args.outdir, args.steps, args.frequency, norm)
    compute_dv_stats(args.outdir)

    return


if __name__ == "__main__":
    main()