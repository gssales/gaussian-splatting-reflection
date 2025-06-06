import sys
from scene import Scene, GaussianModel
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render, network_gui
from utils.image_utils import render_net_image
import torch

def view(dataset, pipe, iteration):
    
    view_render_options = ["RGB", "Depth"]
    if dataset.surfel_splatting:
        view_render_options = ['RGB', 'Alpha', 'Normal', 'Depth']
    if dataset.deferred_reflection:
        view_render_options += ["Base Color", "Refl. Strength", "Normal", "Refl. Color"]

    gaussians = GaussianModel(dataset.sh_degree, dataset.deferred_reflection, dataset.surfel_splatting)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    while True:
        with torch.no_grad():
            if network_gui.conn == None:
                network_gui.try_connect(view_render_options)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)
                        net_image = render_net_image(render_pkg, view_render_options, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "it": iteration,
                        "max": gaussians.get_scaling.max().item()
                        # Add more metrics as needed
                    }
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                except Exception as e:
                    raise e
                    print('Viewer closed')
                    exit(0)

if __name__ == "__main__":

    # Set up command line argument parser
    parser = ArgumentParser(description="Exporting script parameters")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--iteration', type=int, default=30000)
    args = get_combined_args(parser)
    print("View: " + args.model_path)
    print("View: ", args)
    network_gui.init(args.ip, args.port)
    
    view(lp.extract(args), pp.extract(args), args.iteration)

    print("\nViewing complete.")