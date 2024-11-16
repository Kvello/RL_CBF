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
    dummy_lidar[0, 0, 0] = 2.0
    dummy_lidar[0, -1,-1] = 1.0
    # We know this pixel corresponds to the right-bottom corner of the fov
    # So the angles should be (-45, 180) in degrees
    theta = torch.deg2rad(torch.tensor([DummyConfig.horizontal_fov_deg_max]))
    phi = torch.deg2rad(torch.tensor([DummyConfig.vertical_fov_deg_min]))
    dir = torch.tensor([[[torch.cos(theta) * torch.cos(phi), 
                          torch.sin(theta) * torch.cos(phi),
                          torch.sin(phi)]]], device=lidar_cbf.device)
    displacements = lidar_cbf.get_displacements(dummy_lidar)
    assert torch.allclose(displacements, dir)
    dummy_lidar[0, -1, -1] = 2.0
    dummy_lidar[0, DummyConfig.height//2, DummyConfig.width//2] = 1.0
    # We know this pixel corresponds to the center of the fov
    # So the angles should be approx(0, 0) in degrees
    vfov_min = DummyConfig.vertical_fov_deg_min
    vfov_max = DummyConfig.vertical_fov_deg_max
    hfov_min = DummyConfig.horizontal_fov_deg_min
    hfov_max = DummyConfig.horizontal_fov_deg_max
    round_off_err_phi = 0 if DummyConfig.height + 1 % 2 == 0 else -(vfov_max-vfov_min)/(2*DummyConfig.height)
    round_off_err_theta = 0 if DummyConfig.width + 1 % 2 == 0 else (hfov_max-hfov_min)/(2*DummyConfig.width)
    phi = torch.deg2rad(torch.tensor([round_off_err_phi]))
    theta = torch.deg2rad(torch.tensor([round_off_err_theta]))
    dir = torch.tensor([[[torch.cos(theta) * torch.cos(phi), 
                          torch.sin(theta) * torch.cos(phi),
                          torch.sin(phi)]]], device=lidar_cbf.device)
    displacements = lidar_cbf.get_displacements(dummy_lidar)
    assert torch.allclose(displacements, dir,rtol = 1e-2)

    dummy_lidar[0, DummyConfig.height//2, DummyConfig.width//2] = 2.0
    dummy_lidar[0, 0, DummyConfig.width//2] = 1.0
    # This pixel corresponds to the top center of the fov
    # So the angles should be (0, -180) in degrees
    theta = torch.deg2rad(torch.tensor([round_off_err_theta]))
    phi = torch.deg2rad(torch.tensor([DummyConfig.vertical_fov_deg_max]))
    dir = torch.tensor([[[torch.cos(theta) * torch.cos(phi), 
                          torch.sin(theta) * torch.cos(phi),
                          torch.sin(phi)]]], device=lidar_cbf.device)
    displacements = lidar_cbf.get_displacements(dummy_lidar)
    assert torch.allclose(displacements, dir,rtol = 1e-2)