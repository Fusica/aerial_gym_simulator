class control:
    """
    Controller parameters adapted from the old aerial_0803 thrust_bodyrate_control path.

    Input action semantics before scaling:
    [normalized_thrust_minus_one, p_rate, q_rate, r_rate]
    """

    num_actions = 4
    scale_input = [1.0, 4.0, 4.0, 2.0]
    kOmega = [1.0, 1.0, 0.8]
    kOmegaI = [0.0, 0.0, 0.0]
    kOmegaD = [0.0, 0.0, 0.0]
    omegaIntegralLimit = [0.0, 0.0, 0.0]
    torqueLimit = [float("inf"), float("inf"), float("inf")]
    randomize_params = False
