import torch

class LiDARDownsampler(torch.nn.Module):
    def __init__(self,sensor_config,width = 32, height = 8,device = "cuda"):
        super(LiDARDownsampler, self).__init__()
        downsample_factor_width = round(sensor_config.width/width)
        downsample_factor_height = round(sensor_config.height/height)
        self.downsample = torch.nn.MaxPool2d(kernel_size=(downsample_factor_height,
                                                          downsample_factor_width),return_indices=True)
        self.downsampled_width = width
        self.sensor_config = sensor_config
        self.downsampled_height = height
        self.img_width = sensor_config.width
        self.img_height = sensor_config.height
        self.device = device
        self._generate_direction_map()
    def forward(self, x):
        ret, _ = self.downsample(-x)
        ret = -ret
        return ret
    def _generate_direction_map(self):
        vfov_min = self.sensor_config.vertical_fov_deg_min
        vfov_max = self.sensor_config.vertical_fov_deg_max
        hfov_min = self.sensor_config.horizontal_fov_deg_min
        hfov_max = self.sensor_config.horizontal_fov_deg_max
        theta_angles = torch.linspace(hfov_min, hfov_max, self.img_width,device=self.device)
        # Remember that for images, y increases downwards
        phi_angles = torch.linspace(vfov_max, vfov_min, self.img_height,device=self.device)
        theta_angles = torch.deg2rad(theta_angles)
        phi_angles = torch.deg2rad(phi_angles)
        angle_space_p, angle_space_t = torch.meshgrid(phi_angles, theta_angles)
        self.direction_map = torch.stack([torch.cos(angle_space_t)*torch.cos(angle_space_p),
                                        torch.sin(angle_space_t)*torch.cos(angle_space_p),
                                        torch.sin(angle_space_p)], dim = -1)
    def get_displacements(self,lidar_images)->torch.Tensor:
        """
        Downsamples the lidar image and returns the displacements in the vehicle frame
        Assumes ENU coordinate system
        Args:
            lidar_images: torch.Tensor of shape (batch_size, img_height, img_width). The original lidar images
        Returns:
            torch.Tensor of shape (batch_size, height*width)
        """
        downsampled, indices = self.downsample(-lidar_images)
        downsampled = -downsampled
        row_indices = indices // self.img_width
        col_indices = indices % self.img_width
        downsampled_directions = self.direction_map[row_indices, col_indices]
        displacements = downsampled.unsqueeze(-1)*downsampled_directions
        return displacements.reshape(-1, self.downsampled_height*self.downsampled_width, 3)