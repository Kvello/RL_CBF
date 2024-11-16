import isaacgym
import torch
import pytest
class DummyConfig:
    width = 640
    height = 480
    vertical_fov_deg_min = -45
    vertical_fov_deg_max = 45
    horizontal_fov_deg_min = -180
    horizontal_fov_deg_max = 180

@pytest.fixture
def lidar_cbf():
    """Fixture to create a LiDARCBF instance."""
    from RL_CBF.lidar_downsampler.lidar_downsampler import LiDARDownsampler
    return LiDARDownsampler(sensor_config =DummyConfig, width=1, height=1, device="cpu")  # Adjust device as needed

def test_generate_direction_map(lidar_cbf):
    """Tests whether the direction map is generated correctly."""
    direction_map = lidar_cbf.direction_map
    assert direction_map.shape == (480, 640, 3)
    assert torch.allclose(direction_map.norm(dim=-1), torch.ones_like(direction_map[..., 0]))

def test_downsample_forward(lidar_cbf):
    """Tests the forward method's downsampling."""
    dummy_input = torch.rand(2, DummyConfig.height, DummyConfig.width, device=lidar_cbf.device)
    output = lidar_cbf.forward(dummy_input)
    expected_shape = (2, lidar_cbf.downsampled_height, lidar_cbf.downsampled_width)
    assert output.shape == expected_shape

def test_get_displacements(lidar_cbf):
    """Tests the get_displacements method."""
    dummy_lidar = 2*torch.ones(1, DummyConfig.height, DummyConfig.width, device=lidar_cbf.device)
    dummy_lidar[0, 0, 0] = 1.0
    # We know this pixel corresponds to the left-top corner of the fov
    # So the angles should be (45, -180) in degrees
    theta = torch.deg2rad(torch.tensor([DummyConfig.horizontal_fov_deg_min]))
    phi = torch.deg2rad(torch.tensor([DummyConfig.vertical_fov_deg_max]))
    dir = torch.tensor([[[torch.cos(theta) * torch.cos(phi), 
                          torch.sin(theta) * torch.cos(phi),
                          torch.sin(phi)]]], device=lidar_cbf.device)
    displacements = lidar_cbf.get_displacements(dummy_lidar)
    assert torch.allclose(displacements, dir)