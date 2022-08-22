from tqdm import tqdm
import os
from os import listdir, mkdir
from os.path import isfile, join, exists
import numpy as np
import pandas as pd
from bagpy import bagreader
import shutil
from scipy.spatial.transform import Rotation as R

import argparse

renameLabelsS = {'pose.pose.position.x': "x",
                 'pose.pose.position.y': "y",
                 'pose.pose.position.z': "z",
                 'pose.pose.orientation.x': "qx",
                 'pose.pose.orientation.y': "qy",
                 'pose.pose.orientation.z': "qz",
                 'pose.pose.orientation.w': "qw",
                 'twist.twist.linear.x': "u",
                 'twist.twist.linear.y': "v",
                 'twist.twist.linear.z': "w",
                 'twist.twist.angular.x': "p",
                 'twist.twist.angular.y': "q",
                 'twist.twist.angular.z': "r"}

renameLabelsA = {'wrench.force.x': "Fx",
                 'wrench.force.y': "Fy",
                 'wrench.force.z': "Fz",
                 'wrench.torque.x': "Tx",
                 'wrench.torque.y': "Ty",
                 'wrench.torque.z': "Tz"}

def clean_bag(dataDir, outDir, n=500):
    corruptDir = join(dataDir, 'corrupted')
    if not exists(corruptDir):
        mkdir(corruptDir)
    if not exists(outDir):
        os.makedirs(outDir)
    files = [f for f in listdir(dataDir) if isfile(join(dataDir, f))]
    t = tqdm(files, desc="Cleaning", ncols=150, colour="green", postfix={"corrupted:": None})
    for f in t:
        bagFile = join(dataDir, f)
        name = os.path.splitext(f)[0]
        bagDir = join(dataDir, name)
        corrupt = join(dataDir, "corrupted", f)
        if exists(bagFile):
            try:
                traj = traj_from_bag(bagFile, renameLabelsS, renameLabelsA)
                columns = traj.columns
            except:
                t.set_postfix({"corrupted:": f"{f}"})
                os.rename(bagFile, corrupt)
                if exists(bagDir):
                    shutil.rmtree(bagDir)
                continue
        pd.DataFrame(data=traj, columns=columns).to_csv(os.path.join(outDir, name + ".csv"))

def traj_from_bag(path, rds, rda):
    bag = bagreader(path, verbose=False)
    dfs = pd.read_csv(bag.message_by_topic("/rexrov2/pose_gt"))
    dfa = pd.read_csv(bag.message_by_topic("/thruster_input"))
    traj = df_traj(dfs, rds, dfa, rda)
    traj = traj.set_index(np.arange(len(traj)))
    return traj

def resample(df, rd):
    labels = list(rd.keys())
    labels.append('Time')
    df = df.loc[:, labels]
    # relative time of a traj as all the trajs are captured in the same gazebo instance.
    df['Time'] = df['Time'] - df['Time'][0]
    df['Time'] = pd.to_datetime(df['Time'], unit='s').round('ms')
    traj = df.copy()
    traj.index = df['Time']
    traj.rename(columns=rd, inplace=True)
    traj.drop('Time', axis=1, inplace=True)
    traj = traj.resample('ms').interpolate('linear').resample('0.1S').interpolate()
    return traj

def df_traj(dfs, rds, dfa, rda):
    trajS = resample(dfs, rds)
    trajA = resample(dfa, rda)
    quats = trajS.loc[:, ['qx', 'qy', 'qz', 'qw']].to_numpy()
    r = R.from_quat(quats)
    euler = r.as_euler('xyz', False)
    mat = r.as_matrix()

    trajS['roll (rad)'] = euler[:, 0]
    trajS['pitch (rad)'] = euler[:, 1]
    trajS['yaw (rad)'] = euler[:, 2]

    trajS['r00'] = mat[:, 0, 0]
    trajS['r01'] = mat[:, 0, 1]
    trajS['r02'] = mat[:, 0, 2]

    trajS['r10'] = mat[:, 1, 0]
    trajS['r11'] = mat[:, 1, 1]
    trajS['r12'] = mat[:, 1, 2]

    trajS['r20'] = mat[:, 2, 0]
    trajS['r21'] = mat[:, 2, 1]
    trajS['r22'] = mat[:, 2, 2]

    traj = pd.concat([trajS, trajA], axis=1)
    return traj

def parse_arg():
    parser = argparse.ArgumentParser(prog="clean_bags",
                                     description="mppi-tensorflow")

    parser.add_argument('-s', '--steps', type=int,
                        help='number of steps to keep in the bag', default=500)

    parser.add_argument('-o', '--outdir', type=str, default=".",
                        help="output directory for cleaned up bags.")

    parser.add_argument('-d', '--datadir', type=str, default=None,
                        help="dir containing the bags to clean.")

    args = parser.parse_args()
    return args

def main():
    args = parse_arg()
    if args.datadir is not None:
        frames = clean_bag(args.datadir, args.outdir, args.steps)
        return
    print("No datadir provided, nothing to clean")

if __name__ == "__main__":
    main()