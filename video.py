from pathlib import Path
from argparse import ArgumentParser
from utils.render_utils import create_video, create_videos

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--render_path", type=str)
    args = parser.parse_args()

    path = Path(args.render_path)

    n_frames = 200# len(list((path / 'renders').glob('*.png')))
    print(f"Creating video from {n_frames} frames in {path}")
    try:
        create_video(base_dir=path,
                input_dir=path, 
                input_format='rgb',
                out_name='render_traj', 
                video_kwargs={
                  'shape': (800, 800),
                  'codec': 'h264',
                  'codec': 'h264',
                  'fps': 12,
                  'crf': 18,
                },
                num_frames=n_frames)
    except Exception as e:
        print(f"[!] Unable to create video for render path {args.render_path}")
        print(e)