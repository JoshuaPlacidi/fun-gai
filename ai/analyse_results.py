import imageio
import argparse
import os

if __name__ == "__main__":

    # initialise argument parser
    parser = argparse.ArgumentParser(description='This file is for generating ML datasets')

    # set arguments
    parser.add_argument('-r','--results_path', type=str, required=True,
                        help='The path to the directory the results you wish to analyse')

    # extract arguments
    args = parser.parse_args()

    images = []
    filenames = [os.path.join(args.results_path, i) for i in os.listdir(args.results_path) if i.endswith('out.png')]
    for filename in filenames:
        for i in range(100):
            images.append(imageio.imread(filename))
    imageio.mimsave('movie.gif', images)


