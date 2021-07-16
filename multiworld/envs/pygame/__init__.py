import gym
from gym.envs.registration import register
import logging

LOGGER = logging.getLogger(__name__)

_REGISTERED = False


def register_custom_envs():
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    LOGGER.info("Registering multiworld pygame gym environments")
    register(
        id='Point2DLargeEnv-offscreen-v0',
        entry_point='multiworld.envs.pygame.point2d:Point2DEnv',
        kwargs={
            'images_are_rgb': True,
            'target_radius': 1,
            'ball_radius': 1,
            'render_onscreen': False,
        },
    )
    register(
        id='Point2DLargeEnv-onscreen-v0',
        entry_point='multiworld.envs.pygame.point2d:Point2DEnv',
        kwargs={
            'images_are_rgb': True,
            'target_radius': 1,
            'ball_radius': 1,
            'render_onscreen': True,
        },
    )

def create_image_48_pointmass_flappy_bird_train_env_v0():
    from multiworld.core.image_env import ImageEnv

    wrapped_env = gym.make('PointmassFlappyBirdTrainEnv-v0')
    return ImageEnv(
        wrapped_env,
        48,
        init_camera=None,
        transpose=True,
        normalize=True,
        non_presampled_goal_img_is_garbage=False,
    )
def create_image_84_pointmass_flappy_bird_train_env_v0():
    from multiworld.core.image_env import ImageEnv

    wrapped_env = gym.make('PointmassFlappyBirdTrainEnv-v0')
    return ImageEnv(
        wrapped_env,
        84,
        init_camera=None,
        transpose=True,
        normalize=True,
        non_presampled_goal_img_is_garbage=False,
    )
def create_image_48_pointmass_flappy_bird_train_env_v1():
    from multiworld.core.image_env import ImageEnv

    wrapped_env = gym.make('PointmassFlappyBirdTrainEnv-v1')
    return ImageEnv(
        wrapped_env,
        48,
        init_camera=None,
        transpose=True,
        normalize=True,
        non_presampled_goal_img_is_garbage=False,
    )


def create_image_84_point2d_wall_flappy_bird_v0():
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.pygame.point2d import Point2DWallEnv

    kwargs = {
        'action_scale': 0.25,
        'wall_shape': 'flappy-bird',
        'wall_thickness': 0.50,
        'render_target': False,
        'render_size': 84,
        'images_are_rgb': True,
        'sample_realistic_goals': True,
        'norm_order': 2,
        'reward_type': 'vectorized_dense',
    }
    wrapped_env = Point2DWallEnv(**kwargs)
    return ImageEnv(
        wrapped_env,
        84,
        init_camera=None,
        transpose=True,
        normalize=True,
        non_presampled_goal_img_is_garbage=False,
    )

def create_image_84_point2d_wall_flappy_bird_v1():
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.pygame.point2d import Point2DWallEnv

    kwargs = {
        'action_scale': 0.25,
        'wall_shape': 'flappy-bird',
        'wall_thickness': 0.50,
        'render_target': False,
        'render_size': 84,
        'images_are_rgb': True,
        'sample_realistic_goals': True,
        'norm_order': 2,
        'reward_type': 'vectorized_dense',
        'ball_low': (-3.5, -1.5),
        'ball_high': (-3, 0.5),
        'goal_low': (3, -0.5),
        'goal_high': (3.5, 1.5),
    }
    wrapped_env = Point2DWallEnv(**kwargs)
    return ImageEnv(
        wrapped_env,
        84,
        init_camera=None,
        transpose=True,
        normalize=True,
        non_presampled_goal_img_is_garbage=False,
    )

def create_image_84_point2d_wall_flappy_bird_v2():
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.pygame.point2d import Point2DWallEnv

    kwargs = {
        'action_scale': 0.25,
        'wall_shape': 'flappy-bird',
        'wall_thickness': 0.50,
        'render_target': False,
        'render_size': 84,
        'images_are_rgb': True,
        'sample_realistic_goals': True,
        'norm_order': 2,
        'reward_type': 'vectorized_dense',
        'ball_low': (-3.5, -3.0),
        'ball_high': (-3, -0.5),
        'goal_low': (3, 0.5),
        'goal_high': (3.5, 3.0),
    }
    wrapped_env = Point2DWallEnv(**kwargs)
    return ImageEnv(
        wrapped_env,
        84,
        init_camera=None,
        transpose=True,
        normalize=True,
        non_presampled_goal_img_is_garbage=False,
    )

def create_image_84_pointmass_uwall_train_env_small_v0():
    from multiworld.core.image_env import ImageEnv

    wrapped_env = gym.make('PointmassUWallTrainEnvSmall-v0')
    return ImageEnv(
        wrapped_env,
        84,
        init_camera=None,
        transpose=True,
        normalize=True,
        non_presampled_goal_img_is_garbage=False,
    )
def create_image_84_pointmass_uwall_test_env_small_v0():
    from multiworld.core.image_env import ImageEnv

    wrapped_env = gym.make('PointmassUWallTestEnvSmall-v0')
    return ImageEnv(
        wrapped_env,
        84,
        init_camera=None,
        transpose=True,
        normalize=True,
        non_presampled_goal_img_is_garbage=False,
    )

def create_image_84_pointmass_uwall_train_env_big_v0():
    from multiworld.core.image_env import ImageEnv

    wrapped_env = gym.make('PointmassUWallTrainEnvBig-v0')
    return ImageEnv(
        wrapped_env,
        84,
        init_camera=None,
        transpose=True,
        normalize=True,
        non_presampled_goal_img_is_garbage=False,
    )
def create_image_48_pointmass_uwall_train_env_big_v0():
    from multiworld.core.image_env import ImageEnv

    wrapped_env = gym.make('PointmassUWallTrainEnvBig-v0')
    return ImageEnv(
        wrapped_env,
        48,
        init_camera=None,
        transpose=True,
        normalize=True,
        non_presampled_goal_img_is_garbage=False,
    )
def create_image_84_pointmass_uwall_test_env_big_v0():
    from multiworld.core.image_env import ImageEnv

    wrapped_env = gym.make('PointmassUWallTestEnvBig-v0')
    return ImageEnv(
        wrapped_env,
        84,
        init_camera=None,
        transpose=True,
        normalize=True,
        non_presampled_goal_img_is_garbage=False,
    )
def create_image_48_pointmass_uwall_test_env_big_v0():
    from multiworld.core.image_env import ImageEnv

    wrapped_env = gym.make('PointmassUWallTestEnvBig-v0')
    return ImageEnv(
        wrapped_env,
        48,
        init_camera=None,
        transpose=True,
        normalize=True,
        non_presampled_goal_img_is_garbage=False,
    )

def create_image_84_pointmass_uwall_train_env_big_v1():
    from multiworld.core.image_env import ImageEnv

    wrapped_env = gym.make('PointmassUWallTrainEnvBig-v1')
    return ImageEnv(
        wrapped_env,
        84,
        init_camera=None,
        transpose=True,
        normalize=True,
        non_presampled_goal_img_is_garbage=False,
    )
def create_image_48_pointmass_uwall_train_env_big_v1():
    from multiworld.core.image_env import ImageEnv

    wrapped_env = gym.make('PointmassUWallTrainEnvBig-v1')
    return ImageEnv(
        wrapped_env,
        48,
        init_camera=None,
        transpose=True,
        normalize=True,
        non_presampled_goal_img_is_garbage=False,
    )
def create_image_84_pointmass_uwall_test_env_big_v1():
    from multiworld.core.image_env import ImageEnv

    wrapped_env = gym.make('PointmassUWallTestEnvBig-v1')
    return ImageEnv(
        wrapped_env,
        84,
        init_camera=None,
        transpose=True,
        normalize=True,
        non_presampled_goal_img_is_garbage=False,
    )
def create_image_48_pointmass_uwall_test_env_big_v1():
    from multiworld.core.image_env import ImageEnv

    wrapped_env = gym.make('PointmassUWallTestEnvBig-v1')
    return ImageEnv(
        wrapped_env,
        48,
        init_camera=None,
        transpose=True,
        normalize=True,
        non_presampled_goal_img_is_garbage=False,
    )
def create_image_48_pointmass_uwall_test_restricted_env_big_v1():
    from multiworld.core.image_env import ImageEnv

    wrapped_env = gym.make('PointmassUWallTestRestrictedEnvBig-v1')
    return ImageEnv(
        wrapped_env,
        48,
        init_camera=None,
        transpose=True,
        normalize=True,
        non_presampled_goal_img_is_garbage=False,
    )
def create_image_84_pointmass_uwall_test_restricted_env_big_v1():
    from multiworld.core.image_env import ImageEnv

    wrapped_env = gym.make('PointmassUWallTestRestrictedEnvBig-v1')
    return ImageEnv(
        wrapped_env,
        84,
        init_camera=None,
        transpose=True,
        normalize=True,
        non_presampled_goal_img_is_garbage=False,
    )

def create_image_84_pointmass_uwall_train_env_big_v2():
    from multiworld.core.image_env import ImageEnv

    wrapped_env = gym.make('PointmassUWallTrainEnvBig-v2')
    return ImageEnv(
        wrapped_env,
        84,
        init_camera=None,
        transpose=True,
        normalize=True,
        non_presampled_goal_img_is_garbage=False,
    )
def create_image_48_pointmass_uwall_train_env_big_v2():
    from multiworld.core.image_env import ImageEnv

    wrapped_env = gym.make('PointmassUWallTrainEnvBig-v2')
    return ImageEnv(
        wrapped_env,
        48,
        init_camera=None,
        transpose=True,
        normalize=True,
        non_presampled_goal_img_is_garbage=False,
    )
def create_image_84_pointmass_uwall_test_env_big_v2():
    from multiworld.core.image_env import ImageEnv

    wrapped_env = gym.make('PointmassUWallTestEnvBig-v2')
    return ImageEnv(
        wrapped_env,
        84,
        init_camera=None,
        transpose=True,
        normalize=True,
        non_presampled_goal_img_is_garbage=False,
    )
def create_image_48_pointmass_uwall_test_env_big_v2():
    from multiworld.core.image_env import ImageEnv

    wrapped_env = gym.make('PointmassUWallTestEnvBig-v2')
    return ImageEnv(
        wrapped_env,
        48,
        init_camera=None,
        transpose=True,
        normalize=True,
        non_presampled_goal_img_is_garbage=False,
    )
def create_image_48_pointmass_uwall_test_restricted_env_big_v2():
    from multiworld.core.image_env import ImageEnv

    wrapped_env = gym.make('PointmassUWallTestRestrictedEnvBig-v2')
    return ImageEnv(
        wrapped_env,
        48,
        init_camera=None,
        transpose=True,
        normalize=True,
        non_presampled_goal_img_is_garbage=False,
    )

def create_image_84_pointmass_uwall_train_env_big_v3():
    from multiworld.core.image_env import ImageEnv

    wrapped_env = gym.make('PointmassUWallTrainEnvBig-v3')
    return ImageEnv(
        wrapped_env,
        84,
        init_camera=None,
        transpose=True,
        normalize=True,
        non_presampled_goal_img_is_garbage=False,
    )
def create_image_48_pointmass_uwall_train_env_big_v3():
    from multiworld.core.image_env import ImageEnv

    wrapped_env = gym.make('PointmassUWallTrainEnvBig-v3')
    return ImageEnv(
        wrapped_env,
        48,
        init_camera=None,
        transpose=True,
        normalize=True,
        non_presampled_goal_img_is_garbage=False,
    )
def create_image_84_pointmass_uwall_test_env_big_v3():
    from multiworld.core.image_env import ImageEnv

    wrapped_env = gym.make('PointmassUWallTestEnvBig-v3')
    return ImageEnv(
        wrapped_env,
        84,
        init_camera=None,
        transpose=True,
        normalize=True,
        non_presampled_goal_img_is_garbage=False,
    )
def create_image_48_pointmass_uwall_test_env_big_v3():
    from multiworld.core.image_env import ImageEnv

    wrapped_env = gym.make('PointmassUWallTestEnvBig-v3')
    return ImageEnv(
        wrapped_env,
        48,
        init_camera=None,
        transpose=True,
        normalize=True,
        non_presampled_goal_img_is_garbage=False,
    )
def create_image_48_pointmass_uwall_test_restricted_env_big_v3():
    from multiworld.core.image_env import ImageEnv

    wrapped_env = gym.make('PointmassUWallTestRestrictedEnvBig-v3')
    return ImageEnv(
        wrapped_env,
        48,
        init_camera=None,
        transpose=True,
        normalize=True,
        non_presampled_goal_img_is_garbage=False,
    )

register_custom_envs()
