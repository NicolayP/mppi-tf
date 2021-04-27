import argparse
import os
from mppi_tf.scripts.utile import parse_config
import yaml
import numpy as np

def parse_arg():
    parser = argparse.ArgumentParser(prog="mppi-gen-conf",
                                     description="generate a yaml config \
                                     to run experiments")

    parser.add_argument('config', metavar='c', type=str,
                        help='the default environment config file')

    parser.add_argument('task', metavar='t', type=str,
                        help="The taks to run, either static or elipse")

    parser.add_argument('-o', '--output', type=str,
                        help="output prefix, if none given, tmp/ will be used")

    parser.add_argument('-t', '--horizon', type=int,
                        help='prediction horizon')
    
    parser.add_argument('-k', '--samples', type=int,
                        help='the number of samples to use')

    parser.add_argument('-l', '--lam', type=float,
                        help='inverse temperature value')

    parser.add_argument('-g', '--gamma', type=float,
                        help='The confidence in the action sequence. \
                        the closer to lambda, the more confident we are.\
                        the closer to 0 the closer to the ubncontrolled dynamics')

    parser.add_argument('-u', '--upsilon', type=float,
                        help="exploration term")
    
    parser.add_argument('-n', '--noise', type=float,
                        help="the world noise estimation.")
    
    parser.add_argument('-v', "--vel", type=float,
                        help="in case of a eliptic cost: multiplier for\
                        the speed cost")
    parser.add_argument('-p', "--pos", type=float,
                        help="in case of a eliptic cost: multiplier for\
                        the position cost")

    args = parser.parse_args()
    return args


def save_conf(dic, name, prefix=None):
    if prefix:
        save_path = os.path.join(prefix, name)
    else:
        save_path = os.path.join('/tmp/', name)
    stream = open(save_path, 'w')
    yaml.dump(dic, stream)


def main():
    args = parse_arg()
    conf = parse_config(args.config)
    if not (args.task == 'elipse' or args.task == 'static'):
        raise NameError

    task = parse_config(os.path.join('../config/tasks/', args.task + "_task.default.yaml"))
    if args.samples:
        conf["samples"] = args.samples

    if args.horizon:
        conf["horizon"] = args.horizon

    if args.lam:
        conf["lambda"] = args.lam

    if args.gamma:
        conf["gamma"] = args.gamma

    if args.upsilon:
        conf["upsilon"] = args.upsilon

    if args.noise:
        row = np.array(conf["noise"]).shape[0]
        conf["noise"] = (args.noise*np.identity(row)).tolist()
    
    if args.pos:
        task["vel"] = args.vel
    
    if args.pos:
        task["pos"] = args.pos


    save_conf(conf, "config.yaml", args.output)
    save_conf(task, "task.yaml", args.output)

    print("*"*5 + " Config generated " + "*"*5)


if __name__ == "__main__":
    main()