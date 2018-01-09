import pandas as pd
import argparse
import matplotlib.pyplot as plt
import matplotlib
import os.path

font = {'family' : 'normal',
        'size'   : 12}

matplotlib.rc('font', **font)


def plot_metrics(filename):
    df = pd.read_csv(filename)
    print(len(df))

    type_ = []
    for _ in range(int(len(df) / 6)):
        for t in ['test', 'dev']:
            for _ in range(3):
                type_.append(t)

    print(type_)

    df['type'] = type_
    plt.grid()
    for t in ['test']:
        df_test = df[df['type'] == t]
        plt.plot(df_test['epochs'], df_test['F1'], 'o', label='accuracy', markersize=12)
    plt.xlabel('Context length')
    plt.ylabel('Accuracy')
    plt.savefig('metrics.pdf')


def plot(filenames):

    for filename in filenames:
        df = pd.read_csv(filename)

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

        plt.savefig('scores_{}.pdf'.format(os.path.basename(filename)))

    plt.figure()

    for filename in filenames:
        df = pd.read_csv(filename)
        plt.grid()
        plt.plot(df['loss'], label='loss for ctx = {}'.format(os.path.basename(filename)[:-4].replace('logs_', '')))
        plt.plot(df['val_loss'])
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

    plt.savefig('losses.pdf')

    plt.figure()

    for filename in filenames:
        df = pd.read_csv(filename)
        plt.grid()
        plt.plot(df['acc'], label='acc {}'.format(filename))
        plt.plot(df['val_acc'])
        plt.xlabel('Epochs')
        plt.ylabel('Categorical accuracy')
        plt.legend()

    plt.savefig('accuracy.pdf')


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', '--filename', nargs='+', help='Logs filename')
    parser.add_argument('--metric', action='store_true')
    arguments = parser.parse_args()

    if arguments.metric:
        plot_metrics(arguments.filename[0])
    else:
        plot(arguments.filename)


if __name__ == '__main__':
    main()