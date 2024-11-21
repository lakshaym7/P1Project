import openpifpaf
import torch
import argparse
import copy
import logging
import torch.multiprocessing as mp
import csv
from default_params import *
from algorithms import *
from helpers import last_ip
import os
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass


class FallDetector:
    def __init__(self, t=DEFAULT_CONSEC_FRAMES):
        self.consecutive_frames = t
        self.args = self.cli()

    def cli(self):
        parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        openpifpaf.network.Factory.cli(parser)
        openpifpaf.decoder.cli(parser)
        parser.add_argument('--resolution', default=0.4, type=float,
                            help='Resolution prescale factor from 640x480.')
        parser.add_argument('--resize', default=None, type=str,
                            help='Force input image resize. Example WIDTHxHEIGHT.')
        parser.add_argument('--num_cams', default=1, type=int,
                            help='Number of Cameras.')
        parser.add_argument('--video', default=None, type=str,
                            help='Path to the video file.')
        parser.add_argument('--image_dir', default=None, type=str,
                            help='Path to a single image or directory of images.')
        parser.add_argument('--debug', default=False, action='store_true',
                            help='Enable debug messages and autoreload.')
        parser.add_argument('--disable_cuda', default=False, action='store_true',
                            help='Disables CUDA support and runs on CPU.')

        vis_args = parser.add_argument_group('Visualisation')
        vis_args.add_argument('--plot_graph', default=False, action='store_true',
                              help='Plot the graph of features extracted from keypoints of pose.')
        vis_args.add_argument('--joints', default=True, action='store_true',
                              help='Draw joint keypoints on the output image.')
        vis_args.add_argument('--skeleton', default=True, action='store_true',
                              help='Draw skeleton on the output image.')
        vis_args.add_argument('--save_output', default=False, action='store_true',
                              help='Save the result in an output directory.')

        args = parser.parse_args()

        # Configure device
        args.device = torch.device('cpu')
        args.pin_memory = False
        if not args.disable_cuda and torch.cuda.is_available():
            args.device = torch.device('cuda')
            args.pin_memory = True

        openpifpaf.decoder.configure(args)
        openpifpaf.network.Factory.configure(args)

        return args

    def process_images(self, image_dir):
        import cv2
        import matplotlib.pyplot as plt
        import openpifpaf

        if os.path.isfile(image_dir):
            image_paths = [image_dir]
        else:
            image_paths = [
                os.path.join(image_dir, f)
                for f in os.listdir(image_dir)
                if f.endswith(('.jpg', '.png'))
            ]

        processor = openpifpaf.Predictor(checkpoint=self.args.checkpoint)

        for image_path in image_paths:
            print(f"Processing {image_path}")
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not read {image_path}. Skipping.")
                continue

            # Run OpenPifPaf pose estimation using the Predictor object directly
            predictions, confidence_map, fields = processor(image)

            # Visualize results
            if self.args.joints or self.args.skeleton:
                visualizer = openpifpaf.show.Canvas()
                annotated_image = visualizer.annotation_painter.annotation(image, predictions)
                plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.show()

            # Save output if required
            if self.args.save_output:
                output_dir = os.path.join(os.path.dirname(image_path), 'output')
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, os.path.basename(image_path))
                cv2.imwrite(output_path, annotated_image)

    def begin(self):
        if self.args.image_dir:
            self.process_images(self.args.image_dir)
        else:
            print("No image or video input provided. Exiting.")
            return



if __name__ == "__main__":
    start_time = time.time()

    # Instantiate and run the FallDetector
    f = FallDetector()
    f.begin()

    # Record the end time
    end_time = time.time()

    # Calculate the total execution time
    total_execution_time = end_time - start_time
    print(f"Total execution time: {total_execution_time:.2f} seconds")
