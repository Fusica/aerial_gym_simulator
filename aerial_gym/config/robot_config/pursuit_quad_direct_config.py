from aerial_gym.config.robot_config.base_quad_root_link_control_config import (
    BaseQuadRootLinkControlCfg,
)


class PursuitQuadDirectCfg(BaseQuadRootLinkControlCfg):
    class robot_asset(BaseQuadRootLinkControlCfg.robot_asset):
        file = "model.urdf"
        name = "pursuit_quad_direct"
        base_link_name = "base_link"
        disable_gravity = False
        collapse_fixed_joints = False
        fix_base_link = False
        collision_mask = 0
        replace_cylinder_with_capsule = False
        flip_visual_attachments = False
        density = 0.00001
        angular_damping = 0.0
        linear_damping = 0.0
        max_angular_velocity = 20.0
        max_linear_velocity = 40.0
        armature = 0.001

    class control_allocator_config(BaseQuadRootLinkControlCfg.control_allocator_config):
        force_application_level = "root_link"
