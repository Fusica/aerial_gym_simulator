class control:
    """
    Controller parameters adapted from the old aerial_0803 thrust_bodyrate_control path.

    Input action semantics before scaling:
    [normalized_thrust_minus_one, p_rate, q_rate, r_rate]
    """

    num_actions = 4
    # Keep body-rate commands inside the root-link allocator's motor-feasible range.
    scale_input = [1.0, 2.0, 2.0, 0.2]
    kOmega = [1.0, 1.0, 0.8]
    kOmegaI = [0.0, 0.0, 0.0]
    kOmegaD = [0.0, 0.0, 0.0]
    omegaIntegralLimit = [0.0, 0.0, 0.0]
    torqueLimit = [2.4, 2.4, 0.18]
    randomize_params = False
