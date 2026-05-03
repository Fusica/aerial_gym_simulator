from aerial_gym import AERIAL_GYM_DIRECTORY


class _FixedAssetBase:
    num_assets = 1
    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/models/environment_assets/objects"
    min_state_ratio = [0.0] * 13
    max_state_ratio = [0.0] * 13
    collision_mask = 0
    disable_gravity = True
    replace_cylinder_with_capsule = False
    flip_visual_attachments = False
    density = 1000.0
    angular_damping = 0.0
    linear_damping = 0.0
    max_angular_velocity = 0.0
    max_linear_velocity = 0.0
    armature = 0.0
    collapse_fixed_joints = False
    fix_base_link = True
    specific_filepath = None
    color = [178, 178, 178]
    semantic_id = 40
    per_link_semantic = False
    semantic_masked_links = {}
    keep_in_env = True
    place_force_sensor = False
    force_sensor_parent_link = "base_link"
    force_sensor_transform = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    use_collision_mesh_instead_of_visual = False
    use_mesh_materials = False
    override_com = False
    override_inertia = False
    convex_decomposition_from_submeshes = False
    vhacd_enabled = False


class target_quad_asset_params:
    num_assets = 1
    asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/robots/quad"
    file = "model.urdf"
    min_state_ratio = [
        0.75759143,
        0.75759143,
        0.5,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    max_state_ratio = min_state_ratio
    collision_mask = 0
    disable_gravity = False
    replace_cylinder_with_capsule = False
    flip_visual_attachments = False
    density = 0.00001
    angular_damping = 0.0
    linear_damping = 0.0
    max_angular_velocity = 20.0
    max_linear_velocity = 40.0
    armature = 0.001
    collapse_fixed_joints = False
    fix_base_link = False
    specific_filepath = None
    color = [80, 140, 255]
    semantic_id = 30
    per_link_semantic = False
    semantic_masked_links = {}
    keep_in_env = True
    place_force_sensor = False
    force_sensor_parent_link = "base_link"
    force_sensor_transform = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    use_collision_mesh_instead_of_visual = False
    use_mesh_materials = False
    override_com = False
    override_inertia = False
    convex_decomposition_from_submeshes = True
    vhacd_enabled = False


class room_floor_asset_params(_FixedAssetBase):
    file = "pursuit_room_floor_ceiling.urdf"
    min_state_ratio = [0.5, 0.5, -0.01, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    max_state_ratio = min_state_ratio
    semantic_id = 41


class room_ceiling_asset_params(_FixedAssetBase):
    file = "pursuit_room_floor_ceiling.urdf"
    min_state_ratio = [0.5, 0.5, 1.01, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    max_state_ratio = min_state_ratio
    semantic_id = 42


class room_left_wall_asset_params(_FixedAssetBase):
    file = "pursuit_room_x_wall.urdf"
    min_state_ratio = [-0.01, 0.5, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    max_state_ratio = min_state_ratio
    semantic_id = 43


class room_right_wall_asset_params(_FixedAssetBase):
    file = "pursuit_room_x_wall.urdf"
    min_state_ratio = [1.01, 0.5, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    max_state_ratio = min_state_ratio
    semantic_id = 44


class room_top_wall_asset_params(_FixedAssetBase):
    file = "pursuit_room_y_wall.urdf"
    min_state_ratio = [0.5, -0.01, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    max_state_ratio = min_state_ratio
    semantic_id = 45


class room_bottom_wall_asset_params(_FixedAssetBase):
    file = "pursuit_room_y_wall.urdf"
    min_state_ratio = [0.5, 1.01, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    max_state_ratio = min_state_ratio
    semantic_id = 46


class static_cylinder_0_asset_params(_FixedAssetBase):
    file = "pursuit_static_cylinder.urdf"
    min_state_ratio = [0.0, 0.7, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    max_state_ratio = min_state_ratio
    semantic_id = 47
    color = [217, 64, 64]


class static_cylinder_1_asset_params(_FixedAssetBase):
    file = "pursuit_static_cylinder.urdf"
    min_state_ratio = [1.0, 0.3, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    max_state_ratio = min_state_ratio
    semantic_id = 48
    color = [64, 140, 217]
