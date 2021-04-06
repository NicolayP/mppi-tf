import argparse
import os

from tqdm import tqdm

from controller_base import ControllerBase
from cost import getCost
from model_base import ModelBase
from simulation import Simulation
from utile import parse_config, parse_dir, gif_path


def parse_arg():
    parser = argparse.ArgumentParser(prog="mppi",
                                     description="mppi-tensorflow")

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument("--replay", action="store_true",
                       help="replay experience from log dir")

    group.add_argument("--new", action="store_true", help="this flag will \
                       expect a config and task file")

    parser.add_argument("--log_dir", type=str, help="when replay is active\
                        this argument should point to the tensorboard logdir\
                        which experiment should be replayed.")

    parser.add_argument('--config', metavar='c', type=str,
                        help='Controler and env config file')

    parser.add_argument('--task', metavar='t', type=str,
                        help="Task description file")

    parser.add_argument('-r', '--render', action='store_true',
                        help="render the simulation")

    parser.add_argument('-d', '--debug', action='store_true',
                        help="debug flag that will store every saved data \
                        inside a graphs/debug/ folder")

    parser.add_argument('-l', '--log', action='store_true',
                        help="log in tensorboard")

    parser.add_argument('-s', '--steps', type=int,
                        help='number of steps', default=200)

    parser.add_argument('-t', '--train', type=int,
                        help='training step iterations', default=10)

    parser.add_argument('-g', '--gif', type=str, default=None,
                        help="Save all the trajectories in a gif file \
                        (takes a lot of time)")

    parser.add_argument('-f', '--filter', action='store_true',
                        help='filters the control sequence to smooth \
                        out the result')

    args = parser.parse_args()
    return args


def main():
    print("*"*5 + " start mppi controller " + "*"*5)
    args = parse_arg()
    if args.new:
        conf = parse_config(args.config)
    elif args.replay:
        conf, args.task = parse_dir(args.log_dir)


    sim = Simulation(conf["env"], conf['state-dim'], conf['action-dim'],
                     None, args.render)

    model = ModelBase(mass=5, dt=conf["dt"], state_dim=conf['state-dim'],
                      act_dim=conf['action-dim'],
                      name=os.path.splitext(os.path.basename(conf["env"]))[0])

    cost_fc = getCost(args.task, conf['lambda'], conf['gamma'],
                      conf['upsilon'], conf['noise'], conf["horizon"])

    cont = ControllerBase(model, cost_fc, k=conf['samples'],
                          tau=conf["horizon"], dt=conf["dt"],
                          s_dim=conf['state-dim'], a_dim=conf['action-dim'],
                          lam=conf['lambda'], upsilon=conf['upsilon'],
                          sigma=conf['noise'], log=args.log, gif=args.gif,
                          normalize_cost=True, filter_seq=args.filter,
                          config_file=args.config, task_file=args.task,
                          debug=args.debug)

    prev_time = sim.getTime()
    time = sim.getTime()

    for step in tqdm(range(args.steps)):

        x = sim.getState()
        u = cont.next(x)
        while time-prev_time < conf["dt"]:
            x_next = sim.step(u)
            time = sim.getTime()

        prev_time = time
        cont.save(x, u, x_next)

        if step % args.train == 0:
            cont.train()

    gif_path(args.steps, args.gif)

    print("*"*5 + " Done " + "*"*5)

def plot_sgf():
    args = parse_arg()
    conf = parse_config(args.config)

    sim = Simulation(conf["env"], conf['state-dim'], conf['action-dim'],
                     None, False)

    model = ModelBase(mass=5,
                      dt=conf["dt"],
                      state_dim=conf['state-dim'],
                      act_dim=conf['action-dim'],
                      name=os.path.splitext(os.path.basename(conf["env"]))[0])

    cost_fc = getCost(args.task, conf['lambda'], conf['gamma'],
                      conf['upsilon'], conf['noise'], conf["horizon"])

    cont = ControllerBase(model, cost_fc, k=conf['samples'],
                          tau=conf["horizon"], dt=conf["dt"],
                          s_dim=conf['state-dim'], a_dim=conf['action-dim'],
                          lam=conf['lambda'], sigma=conf['noise'],
                          log=args.log, config_file=args.config)

    x = sim.getState()
    _ = cont.next(x)


if __name__ == '__main__':
    main()
