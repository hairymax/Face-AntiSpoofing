import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize
import numpy as np

def roc_curve_plots(y_true, models_proba, title='ROC Curve', figsize=(12, 9)):
    ''' Функция построения ROC кривой для моделей из переданнойго словаря

        Принимает:
    '''
    plt.figure(figsize=figsize)
    for name, proba in models_proba.items():
        fpr, tpr, thresholds = roc_curve(y_true, proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label='AUC = %0.4f (%s)' % (roc_auc, name))
    plt.title(title)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.grid()
    #plt.xlim([0, 1]), plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def multiclass_roc_curve_plots(y_true, proba, class_labels, title='ROC Curve', figsize=(12, 9)):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    y_true = label_binarize(y_true, classes=[0,1,2])
    n_classes = y_true.shape[1]
    
    plt.figure(figsize=figsize)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], 
                 label="Class: {0} (AUC = {1:0.4f})".format(class_labels[i], roc_auc[i]),)
    plt.title(title)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.grid()
    #plt.xlim([0, 1]), plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
def confusion_matricies_plots(confusion_matricies, figsize=(12,4)):
    import pandas as pd
    _, ax =plt.subplots(1, len(confusion_matricies), figsize=figsize)
    i = 0
    for name, matrix in confusion_matricies.items():
        sns.heatmap(pd.DataFrame(matrix), annot=True, annot_kws={"fontsize":12}, 
                    cmap='Blues', vmin=0, vmax=1, ax=ax[i])
        ax[i].set(title=name, xlabel='Predictions', ylabel='True')
        i += 1
    plt.suptitle('Confusion matricies', fontsize=12, y=1.1)
    plt.show()

def plot_value_counts(series, n_values=25, fillna='NONE', figwidth=12, 
                      bar_thickness=0.5, sort_index=False,
                      verbose=False, show_percents=False):
    ''' Визуализация количества встречающихся значений в pd.Series

    Параметры
    ---
    `series` : pd.Series
    `column` : str - название столбца
    `n_values` : int - максимальное количество значений для отображения на диаграмме
    `fillna` : Any - значение, которым необходимо заполнить пропуски
    `verbose`: bool - показывать ли уникальные значения
    `show_percents`: bool - показывать долю значений в процентах
    '''

    _ = series.dropna().unique()
    if verbose:
        print('`{}`, {} unique values: \n{}'.format(series.name, len(_), sorted(_)))

    val_counts = series.fillna(fillna).value_counts()
    if sort_index:
        val_counts = val_counts.sort_index()
    bar_values = val_counts.values[:n_values]
    bar_labels = val_counts.index[:n_values].astype('str')
    plt.figure(figsize=(figwidth, bar_thickness * min(len(val_counts), n_values)))
    ax = sns.barplot(x=bar_values, y=bar_labels)
    ax.set(title='"{}" value counts ({} / {})'
           .format(series.name, len(bar_labels), val_counts.shape[0]),
           #xlim=[0, 1.07*bar_values.max()]
           )
    if show_percents:
        labels = [f'{w/val_counts.values.sum()*100:0.1f}%' 
                    if (w := v.get_width()) > 0 else '' for v in ax.containers[0]]
    else:
        labels = bar_values
    plt.bar_label(ax.containers[0], labels=labels, label_type='center')
    for i in range(len(bar_labels)):
        if bar_labels[i] == fillna:
            ax.patches[i].set_color('black')
    plt.grid()
    plt.show()
