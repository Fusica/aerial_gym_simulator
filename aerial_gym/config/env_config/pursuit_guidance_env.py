from aerial_gym.config.asset_config.pursuit_guidance_asset_config import (
    room_bottom_wall_asset_params,
    room_ceiling_asset_params,
    room_floor_asset_params,
    room_left_wall_asset_params,
    room_right_wall_asset_params,
    room_top_wall_asset_params,
    static_cylinder_0_asset_params,
    static_cylinder_1_asset_params,
    target_quad_asset_params,
    target_x500_asset_params,
)


class PursuitGuidanceEnvCfg:
    class env:
        num_envs = 512
        num_env_actions = 0
        env_spacing = 1800.0
        num_physics_steps_per_env_step_mean = 1
        num_physics_steps_per_env_step_std = 0
        render_viewer_every_n_steps = 10
        collision_force_threshold = 0.1
        manual_camera_trigger = False
        reset_on_collision = True
        create_ground_plane = False
        sample_timestep_for_latency = False
        perturb_observations = False
        keep_same_env_for_num_episodes = 1
        write_to_sim_at_every_timestep = False
        use_warp = False

        lower_bound_min = [0.0, 0.0, 0.0]
        lower_bound_max = [0.0, 0.0, 0.0]
        upper_bound_min = [1000.0, 1000.0, 1000.0]
        upper_bound_max = [1000.0, 1000.0, 1000.0]

    class env_config:
        include_asset_type = {
            "room_floor": True,
            "room_ceiling": True,
            "room_left_wall": True,
            "room_right_wall": True,
            "room_top_wall": True,
            "room_bottom_wall": True,
            "static_cylinder_0": True,
            "static_cylinder_1": True,
            "target_quad": False,
            "target_x500": True,
        }
        asset_type_to_dict_map = {
            "room_floor": room_floor_asset_params,
            "room_ceiling": room_ceiling_asset_params,
            "room_left_wall": room_left_wall_asset_params,
            "room_right_wall": room_right_wall_asset_params,
            "room_top_wall": room_top_wall_asset_params,
            "room_bottom_wall": room_bottom_wall_asset_params,
            "static_cylinder_0": static_cylinder_0_asset_params,
            "static_cylinder_1": static_cylinder_1_asset_params,
            "target_quad": target_quad_asset_params,
            "target_x500": target_x500_asset_params,
        }
