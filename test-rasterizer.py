
from argparse import ArgumentParser
import sys
import traceback

import torch

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import network_gui, render
from scene import StageScene
from scene.gaussian_model import GaussianModel
from utils.general_utils import safe_state
from utils.image_utils import render_net_image

def stage(dataset: ModelParams, opt: OptimizationParams, pipe: PipelineParams):
  
  gaussians = GaussianModel(dataset.sh_degree, opt.refl_init_value, dataset.cubemap_resol)
  scene = StageScene(dataset, gaussians)
  gaussians.training_setup(opt)
  
  bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
  background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
  
  view_render_options = ['RGB', 'Alpha', 'Normal', 'Depth', "Base Color", "Refl. Strength", "Normal", "Refl. Color", "Edge", "Curvature", "Mask"]
  while network_gui.conn == None:
    network_gui.try_connect(view_render_options)
  while network_gui.conn != None:
    try:
      net_image_bytes = None
      custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
      if custom_cam != None:
        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer, initial_stage=False, env_scope_center=opt.env_scope_center, env_scope_radius=opt.env_scope_radius)   
        net_image = render_net_image(render_pkg, view_render_options, render_mode, custom_cam)
        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
      metrics_dict = {
        "#": gaussians.get_opacity.shape[0],
        # Add more metrics as needed
      }
      # Send the data
      network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
    except Exception as e:
      # raise e
      traceback.print_exc()
      print(e)
      network_gui.conn = None

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    args = parser.parse_args(sys.argv[1:])

    print("\nTesting simple Gaussians")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    stage(lp.extract(args), op.extract(args), pp.extract(args))

    # All done
    print("\nStaging complete.")