from re import L
from .controllers.controller_base import ControllerBase
from .controllers.lagged_controller import LaggedModelController
from .controllers.state_controller import StateModelController
from .controllers.dmd_controller import DMDMPPI


def dmd(
        model, cost,
        k, tau,
        sDim, aDim,
        lam, upsilon, sigma,
        normalizeCost, filterSeq,
        log, logPath,
        graphMode, debug,
        configDict, taskDict, modelDict
    ):
    pass

def lagged(
        model, cost,
        k, tau,
        sDim, aDim,
        lam, upsilon, sigma, initSeq,
        normalizeCost, filterSeq,
        log, logPath,
        graphMode, debug,
        configDict, taskDict, modelDict
    ):
    return LaggedModelController(
        model=model, cost=cost,
        k=k, tau=tau,
        sDim=sDim, aDim=aDim, h=configDict['history'],
        lam=lam, upsilon=upsilon, sigma=sigma, initSeq=initSeq,
        normalizeCost=normalizeCost, filterSeq=filterSeq,
        log=log, logPath=logPath,
        graphMode=graphMode, debug=debug,
        configDict=configDict, taskDict=taskDict, modelDict=modelDict
    )

def state(
        model, cost,
        k, tau,
        sDim, aDim,
        lam, upsilon, sigma, initSeq,
        normalizeCost, filterSeq,
        log, logPath,
        graphMode, debug,
        configDict, taskDict, modelDict
    ):
    return StateModelController(
        model=model, cost=cost,
        k=k, tau=tau,
        sDim=sDim, aDim=aDim,
        lam=lam, upsilon=upsilon, sigma=sigma, initSeq=initSeq,
        normalizeCost=normalizeCost, filterSeq=filterSeq,
        log=log, logPath=logPath,
        graphMode=graphMode, debug=debug,
        configDict=configDict, taskDict=taskDict, modelDict=modelDict
    )

def get_controller(
    model, cost,
    k, tau,
    sDim, aDim,
    lam, upsilon, sigma,
    initSeq,
    normalizeCost, filterSeq,
    log, logPath,
    graphMode, debug,
    configDict, taskDict, modelDict):

    switcher = {
        "dmd_controller": dmd,
        "lagged_controller": lagged,
        "state_controller": state
    }

    controller_type = configDict['type']
    getter = switcher.get(controller_type, lambda: "invalid controller type, check\
        spelling, supported are: ")

    return getter(
        model=model, cost=cost, k=k, tau=tau, sDim=sDim, aDim=aDim,
        lam=lam, upsilon=upsilon, sigma=sigma, initSeq=initSeq, normalizeCost=normalizeCost,
        filterSeq=filterSeq, log=log, logPath=logPath, graphMode=graphMode, debug=debug,
        configDict=configDict, taskDict=taskDict, modelDict=modelDict
    )
