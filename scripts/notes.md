## Notes to myself during implementation.

# 1. 
    It seems like computing fosses enquation by merging all the elementes related to the rigid body and the added mass as in vehicle.py doesnt seem to work.
    
    In gazebo they compute the acceleration first from the previous step so with a timesift of -1. The using this acceleration they compute the added mass effect from it and it thus seems like gazebo is only computing the acceleration of the rigid body to compute the next step ? Might need to look at the ode solver. 

    Next try is to compute the acceleration from the last step and add this to the rhs of the equation. Hopes this will shift the acceleration closer to gazebo's acceleration.

    Implemented the runge kutta method, seems more stable.

    IDEA:
        - wrong integration for quaternions? Need to check the maths. Integration plus combination. 
        Note: final rotation represented with quaternion. To find next state need to multiply quaterinions and not add them.!!!