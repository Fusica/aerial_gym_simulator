import math


class task_config:
    seed = 1
    sim_name = "base_sim"
    env_name = "pursuit_guidance_env"
    robot_name = "pursuit_quad_direct"
    controller_name = "thrust_bodyrate_control"
    args = {}
    num_envs = 512
    use_warp = False
    headless = False
    device = "cuda:0"
    observation_space_dim = 32
    privileged_observation_space_dim = 0
    action_space_dim = 4
    episode_len_steps = 3600
    return_state_before_reset = False

    class room:
        enabled = True
        size = [1000.0, 1000.0, 1000.0]
        wall_thickness = 20.0
        wall_color = [0.7, 0.7, 0.7]

    class static_obstacles:
        enabled = True
        positions = [
            [0.0, 700.0, 500.0],
            [1000.0, 300.0, 500.0],
        ]
        quaternions = [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        colors = [
            [0.85, 0.25, 0.25],
            [0.25, 0.55, 0.85],
        ]
        radius = 300.0

    class reset:
        actor1_position = [755.0, 750.0, 510.0]
        actor1_linear_velocity = [-25.0, -10.0, 0.0]
        actor1_angular_velocity = [0.0, 0.0, 0.0]
        actor1_quaternion = [0.0, 0.0, 0.0, 1.0]

        actor2_position = [757.59143, 757.59143, 500.0]
        actor2_linear_velocity = [-5.43323, 5.43323, 0.0]
        actor2_angular_velocity = [0.0, 0.0, 0.0]
        actor2_quaternion = [0.0, 0.0, 0.9238795, 0.38268343]

    class control2:
        controller = "lee_position_control"
        kP = [0.8, 0.8, 1.0]
        kV = [0.5, 0.5, 0.4]
        kR = [4.0, 4.0, 4.0 / 3.0]
        kOmega = [0.5, 0.5, 1.2]
        scale_input = [1.0, 1.0, 1.0, 1.0]

    class target_motion:
        type = "apf_escape"
        random_choices = ["simple_linear", "sine_linear", "elliptical"]
        validation_samples = 65
        max_sample_attempts = 128
        wall_margin = 50.0
        obstacle_margin = 50.0

        class simple_linear:
            speed = 20.0
            direction = [-14.14, -14.14, 0.5]

            class randomization:
                start_position_delta = [25.0, 25.0, 10.0]
                direction_delta = [0.04, 0.04, 0.01]
                speed_delta = 2.0

        class sine_linear:
            speed = 20.0
            direction = [-14.14, -14.14, 0.5]
            amp_y = 60.0
            omega_y = 0.41887902047863906
            amp_z = 0.5
            omega_z = 0.05

            class randomization:
                start_position_delta = [18.0, 18.0, 6.0]
                direction_delta = [0.03, 0.03, 0.008]
                speed_delta = 1.5
                amp_y_delta = 10.0
                omega_y_delta = 0.06
                amp_z_delta = 0.5
                omega_z_delta = 0.01
                phase_y_delta = 0.35
                phase_z_delta = 0.20

        class elliptical:
            focus1 = [150.0, 150.0]
            focus2 = [750.0, 750.0]
            semi_major_axis = 435.0
            angular_rate = 0.08
            height = 500.0
            initial_phase = 0.0

            class randomization:
                focus1_delta = [15.0, 15.0]
                focus2_delta = [15.0, 15.0]
                semi_major_axis_delta = 10.0
                angular_rate_delta = 0.008
                height_delta = 10.0
                initial_phase_delta = 0.25
                randomize_rotation_direction = False

        class apf_escape:
            baseline_speed = 20.0
            baseline_direction = [-14.14, -14.14, 0.5]
            baseline_weight = 1.0
            evasive_baseline_weight = 0.5
            response_distance = 100.0
            response_width = 20.0
            min_escape_ratio = 0.0
            max_escape_ratio = 1.0
            max_speed = 24.0
            max_accel = 8.0
            velocity_smoothing = 0.55
            command_lookahead = 0.45
            pursuer_repulsion_weight = 0.55
            pursuer_tangent_weight = 0.35
            wall_influence_distance = 160.0
            wall_repulsion_weight = 1.0
            obstacle_influence_distance = 220.0
            obstacle_repulsion_weight = 1.2

            class randomization:
                start_position_delta = [20.0, 20.0, 8.0]
                baseline_direction_delta = [0.04, 0.04, 0.01]
                baseline_speed_delta = 2.0

    class reward:
        eps = 1e-6
        weight_progress = 0.5
        success_threshold = 1.0
        success_forward_alignment_cos = 0.95
        base_success_bonus = 20.0
        early_bonus_scale = 10.0
        time_penalty_per_step = 0.005
        timeout_penalty = 6.0
        smooth_penalty_coef = -0.001
        amplitude_penalty_coef = -0.0001
        weight_alignment = 0.005
        weight_success_bonus = 0.15
        weight_effort = 1.0
        weight_avoid = 0.15
        far_terminate_distance = 1000.0
        collision_force_threshold = 0.1
        collision_penalty_actor1 = 20.0
        terminate_on_collision = True

    class safety:
        wall_hard_margin = 50.0
        obstacle_hard_margin = 50.0
        avoid_margin = 60.0
        static_obstacle_radius = 300.0
