from .learners.learner_base import LearnerBase


def get_learner(model,
                filename=None,
                bufferSize=264,
                numEpochs=100,
                batchSize=30,
                log=False,
                logPath=None):

    return LearnerBase(model,
                       filename=filename,
                       bufferSize=bufferSize,
                       numEpochs=numEpochs,
                       batchSize=batchSize,
                       log=log,
                       logPath=logPath)
