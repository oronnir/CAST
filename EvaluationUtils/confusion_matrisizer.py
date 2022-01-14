import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import itertools
import time


def plot_cm(cm, target_names, title, cmap=None, normalize=True, save_figure_path=None):
    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    plt.rc('figure', titlesize=22)
    matplotlib.rcParams.update({'font.size': 16})
    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(12, 11))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            text_format = "{:0.3f}" if cm.shape[0] < 10 else "{:.2f}"
            plt.text(j, i, text_format.format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\nAccuracy={:0.4f}'.format(accuracy))
    if save_figure_path:
        fig_ap = plt.gcf()
        fig_ap.savefig(save_figure_path, format='png', dpi=100, bbox_inches='tight')
        time.sleep(2)
    plt.show()
    time.sleep(4)
    plt.close('all')
