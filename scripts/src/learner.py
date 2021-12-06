from .learners.learner_base import LearnerBase


def get_learner(model, log=False, logPath=None):
    return LearnerBase(model, log=log, logPath=logPath)
