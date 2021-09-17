# NOTE:

## Constant input in body frame

### X:

*POSE:*
    The shape of the graph seems realtively correct. There is a small difference in the angles prediction but the max is in the pitch ~5degrees.
*VELOCITY:*
    The different velocities are matching for X, Y and Z. Not at the end of Z but seems to be related to the pitch angle velocity becoming negative and thus converts some forward velocity to some z vel. Again there seeems to be a difference in the angluar velocity. (not in Yaw)
*ACCELERATION:*
    Acceleration profile seems correct apart from the angular acceleration (not in Yaw)
    This model should be useable by MPPI to control the X state.


### Y:
*POSE:*
    The shape of seems okay-ish for X, Y, Yaw and pitch (not the same but small difference) There seems to be a high roll value. This will need some identification because this force shouldn't generate any torque as it is applied to the COG.
    The z difference seems again to be related the roll value.
*VELOCITY:*
    Wiredly enough, the z position is going to a positive value despite the velocity in body frame beeing negative. It could be explained by the observed roll angle.
*ACCELERATION:*
    The overall accleration seems close to the gazebo one (except for the usuall oscilliation) appart from the z axis. Needs investigation.

### Z:
*POSE:*
    The overall pose graph seems correct. appart from the pitch and roll but only |3| degrees.
*VELOCITY:*
    The velocity again seems okay appart from the usual oscillliation on the angular velocity.
*ACCELERATION:*
    The acceleration profile seems correct with the observed pose/velocity profile.

### P:
*POSE:*
    The Roll profile seems better in my model than in the gazebo simulation where the angle stays to 0 despite receiving only a roll torque. The obsersved variation in Y is probably due to the roll. The others deviate from gazebo but again only by a small margin.
*VELOCITY:*
    The velocity profiles are very different from gazebo but at least the velocity in roll seems correct.
*ACCELERATION:*
    The acceleration profiles seems to damp out to 0. (as expected?)

### Q:
*POSE:*
    The profile seems correct up to the point where pitch is reaching 90 degrees. At this point roll and ptich rotate by 180 degrees (seems correct) but the pitch is decreasing constantly. This is probably due to some tangant computation somewhere.
    It seems like for the previous graph, gazebo is not computing the angular velocity correctly? No change in pitch.
*VELOCITY:*
    The Pitch velocity seems correct. I can't really visualize how pitch only input force will affect the other DOF. Need to double check that!
*ACCELERATION:*
    Acceleration profile seems correctly to damp out towards 0.

### R:
*POSE:*
    The profiles seems to be the closest to gazebo. This time the Yaw angle seems to be correct in gazebo. As always there is a small difference in the angles (roll and pitch).
*VELOCITY:*
    The velocity profiles differ drastically, I'm woundering if gazebo's velocity is expressed in body frame of world frame. This requires investigation. Need to check why roll and pitch are not moving at all.
*ACCELERATION:*
    It seems like the accelerations are converging towrds 0.

Corrected the model according to comments from discord. The vehicle model in uuvsim applied the cross product on the restoring forces to compute the moments before applieingto rotation to the body frame. 
