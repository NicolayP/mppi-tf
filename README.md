# Torch implementation of MPPI

## Dependencies:

## Installation:

## Running:

To run the model for one step, call the main file:

```bash
python run_controller.py
```

The controller works as follows:

- Instanciate a *model*, a *cost*, a *controller* and a *observer*.
- Call the controller on a fake state.