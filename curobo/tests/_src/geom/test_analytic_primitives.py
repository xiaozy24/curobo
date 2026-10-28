"""Analytic primitives survive scene conversion and all cache-loading paths."""
import pytest
import torch

from curobo._src.geom.types import Cuboid, Cylinder, SceneCfg, Sphere
from curobo._src.geom.data.data_cuboid import primitive_parameters
from curobo._src.geom.collision.collision_scene import SceneCollisionCfg, create_scene_collision
from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.util.warp import init_warp

POSE = [0., 0., 0., 1., 0., 0., 0.]


def test_scene_conversion_preserves_analytic_shapes():
    scene = SceneCfg(cuboid=[Cuboid(name='box', pose=POSE, dims=[.1]*3)],
                     sphere=[Sphere(name='ball', pose=POSE, radius=.05)],
                     cylinder=[Cylinder(name='cylinder', pose=POSE, radius=.04, height=.12)])
    supported = scene.get_collision_check_world()
    assert supported.get_cache_dict() == {'obb': 3, 'mesh': 0}
    assert len(supported.sphere) == len(supported.cylinder) == len(supported.cuboid) == 1
    assert supported.sphere[0].radius == .05
    assert supported.cylinder[0].height == .12


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('batched', [False, True])
def test_initial_scene_and_batch_caches_include_spheres(cuda_device_cfg, batched):
    device = cuda_device_cfg
    init_warp(quiet=True, device_cfg=device)
    ball = SceneCfg(sphere=[Sphere(name='ball', pose=POSE, radius=.05)])
    cylinder = SceneCfg(cylinder=[Cylinder(name='cylinder', pose=POSE, radius=.1, height=.12)])
    checker = create_scene_collision(SceneCollisionCfg(
        device_cfg=device, scene_model=[ball, cylinder] if batched else ball))
    # This point lies in the former bounding cube, outside the actual sphere.
    points = [[.045, .045, 0., .002], [.045, 0., 0., .002]]
    q = device.to_device(points).reshape(1, 1, 2, 4)
    envs = None
    if batched:
        q = q.repeat(2, 1, 1, 1)
        envs = torch.tensor([0, 1], dtype=torch.int32, device=device.device)
    buffer = CollisionBuffer.from_shape(q.shape, device)
    cost = checker.checker.get_sphere_distance(checker.data, q, buffer,
        device.to_device([1.]), device.to_device([0.]), env_query_idx=envs)
    torch.testing.assert_close(cost[0, 0], device.to_device([0., .007]), atol=1e-7, rtol=0)
    if batched:
        # At z=0 the radial cylinder wall is nearer than its end cap.
        expected = device.to_device([.102-(2*.045**2)**.5, .057])
        torch.testing.assert_close(cost[1, 0], expected, atol=1e-7, rtol=0)


@pytest.mark.parametrize('radius', [-.1, 0., float('nan'), float('inf')])
def test_invalid_sphere_radius_is_rejected(radius):
    with pytest.raises(ValueError):
        primitive_parameters(Sphere(name='ball', pose=POSE, radius=radius))
