import pandas as pd
import argparse
import matplotlib.pyplot as plt
import matplotlib

font = {'family' : 'normal',
        'size'   : 8}

matplotlib.rc('font', **font)


def plot(filenames):

    for filename in filenames:
        df = pd.read_csv(filename)

        plt.figure()
        f, ax_array = plt.subplots(3, 3, sharex=True, sharey=False)

        i = 0
        for name in df.columns.values:
            if name.startswith('f1'):
                print(int(i / 3), i % 3, name[3:])
                ax = ax_array[int(i / 3), (i % 3)]
                ax.plot(df[name])
                name = name[4:]
                ax.plot(df['precision: {}'.format(name)])
                ax.plot(df['recall: {}'.format(name)])
                ax.set_title(name)
                i += 1

        plt.savefig('scores.pdf')

    plt.figure()

    for filename in filenames:
        df = pd.read_csv(filename)
        plt.plot(df['loss'])
        plt.plot(df['val_loss'])
        plt.legend()
        plt.savefig('losses.pdf')


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', '--filename', nargs='+', help='Logs filename')
    arguments = parser.parse_args()

    plot(arguments.filename)


if __name__ == '__main__':
    main()