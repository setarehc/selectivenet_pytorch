from argparse import ArgumentParser
import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main(args):
    plot(args)

def plot(args):
    """
    reference
    - https://qiita.com/ryo111/items/bf24c8cf508ad90cfe2e (how to make make block)
    - https://heavywatal.github.io/python/matplotlib.html
    """

    if (args.plot_all is True) and (args.plot_test is True):
        raise ValueError('invalid option. either "plot_all" or "plot_test" should be True.')

    # load csv file and plot
    df = pd.read_csv(args.target_path)

    # plot all variable. this is basically used for visualize training log.
    if args.plot_all:
        # ignore some columns
        ignore_columns = ['Unnamed: 0', 'time stamp', 'step', args.x]
        column_names = [column for column in df.columns if column not in ignore_columns]
        
        # create figure
        fig = plt.figure(figsize=(4*len(column_names),3))

        for i, column_name in enumerate(column_names):
            ax = fig.add_subplot(1, len(column_names), i+1)
            sns.lineplot(x=args.x, y=column_name, ci="sd", data=df)
                
        plt.tight_layout()

    # plot test.csv file. 
    elif args.plot_test:
        # ignore some columns
        ignore_columns = ['Unnamed: 0', 'time stamp', 'path', 'loss', 'selective_loss', args.x]
        column_names = [column for column in df.columns if column not in ignore_columns]

        # create figure
        fig = plt.figure(figsize=(4*len(column_names),3))

        for i, column_name in enumerate(column_names):
            ax = fig.add_subplot(1, len(column_names), i+1)
            sns.lineplot(x=args.x, y=column_name, ci="sd", data=df)
                
        plt.tight_layout()

    # plot specified variable
    else:
        if args.y == '':
            raise ValueError('please specify "y"')
        fig = plt.figure()
        ax = fig.subplots()
        sns.lineplot(x=args.x, y=args.y, ci="sd", data=df)
    
    # show and save
    if args.save:
        plt.close()
        if args.log_path == '': 
            raise ValueError('please specify "log_path"')
        os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
        fig.savefig(args.log_path)
    else:
        plt.show()
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-t', '--target_path', type=str, required=True)
    parser.add_argument('-x', type=str, required=True)
    parser.add_argument('-y', type=str, default='')
    parser.add_argument('--plot_all', action='store_true', help='plot all in single image')
    parser.add_argument('--plot_test', action='store_true', help='plot test.csv file')
    parser.add_argument('-l', '--log_path', type=str, default='', help='path of log')
    parser.add_argument('-s', '--save', action='store_true', help='save results')
    args = parser.parse_args()
    main(args)