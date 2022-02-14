import tensorflow as tf
gpu = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(device=gpu[0], enable=True)

from src.model import get_model
import argparse as ap
from src.misc.test_models import test_models
import yaml
import os
import re
import sys

if __name__ == "__main__":
    parser = ap.ArgumentParser(description="Plots the trajectory generated by \
                                           different model for a given action \
                                           sequence file and model configs.",
                               formatter_class=ap.ArgumentDefaultsHelpFormatter)

    parser.add_argument('actionSequence',
                        help="File containing the action sequence to apply. \
                             The sequence should be formated readable by numpy.")

    parser.add_argument("groundTruth",
                        help="file containing the ground truth trajectory \
                             from the robot/simulator in a numpy compatible format.")

    parser.add_argument("modelsConfig",
                        nargs='+',
                        help="A sequence of config files/directories \
                              containing the  models desctiptions. All \
                              the models will be propagated plotted. \
                              If the argument is a file, it will \
                              considered as a config file and will \
                              load the model accordingly. If it's a \
                              directory, it will be considered as \
                              learning log and will load the model \
                              config as well as the latest model param.")

    parser.add_argument('-t',
                        '--time',
                        default=20,
                        help="Define the time interval of the simulation \
                             in seconds.")

    args = parser.parse_args()

    models = []
    labels = []
    for conf in args.modelsConfig:
        if os.path.isfile(conf):
            paramPath = None
            with open(conf, "r") as stream:
                modelDict = yaml.load(stream)
        elif os.path.isdir(conf):
            maxStep = 0
            path = os.path.join(conf, "learner")
            paramDirs = os.listdir(os.path.join(conf, "learner"))
            for paramDir in paramDirs:
                if paramDir.startswith("weights_step"):
                    name = paramDir.split('/')[0]
                    step = int(re.findall(r'\d+', name)[0])
                    if step > maxStep:
                        paramPath = os.path.join(path, paramDir)
                        maxStep = step

            modelFile = os.path.join(conf, "controller", "model.yaml")
            with open(modelFile, "r") as stream:
                modelDict = yaml.load(stream)

        models.append(get_model(modelDict,
                                tf.Variable(1),
                                0.1,
                                13,
                                6,
                                modelDict['type'],
                                paramPath))
        labels.append(modelDict['type'])

    test_models(args.actionSequence,
                args.groundTruth,
                models,
                labels,
                args.time)