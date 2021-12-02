from .src.learners.learner_base import LearnerBase


def get_learner(model):
    return LearnerBase(model)
