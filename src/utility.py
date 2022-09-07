import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize
import numpy as np


def spoof_labels_to_classes(labels_df, classes):
    df = labels_df.copy()
    if len(classes) == 2:
        def to_class(v):
            return classes[0] if v == 0 else classes[1]
    elif len(classes) == 3:
        def to_class(v):
            if v in [1,2,3]:
                return classes[1]
            if v in [7,8,9]:
                return classes[2]
            if v == 0:
                return classes[0]
            return None
    else:
        print('Labels will not be changed')
        def to_class(v):
            return v
    df.iloc[:,1] = df.iloc[:,1].apply(lambda s: to_class(s))   
    
    return df

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


def plot_iter_images(iter, size, count):
    import torchvision.transforms as T
    
    rows = count // 4
    if len(iter) > 2:
        sample, ft_sample, target = iter
    else:
        sample, target = iter
        ft_sample = None
    target = target.numpy()
    fig = plt.figure(figsize=(12, 4*rows))
    for i in range(count):
        ax = fig.add_subplot(rows, 4, i+1)
        ax.axis('off')
        
        plt.imshow(T.ToPILImage()(sample[i]), extent=(0,size,0,size))
        plt.text(0, -20, target[i], fontsize = 20, color='red')
        if ft_sample is not None:
            plt.imshow(T.ToPILImage()(ft_sample[i]), 
                       extent=(3*size/4,5*size/4,-size/4,size/4))
        plt.xlim(0, 5*size/4)
        plt.ylim(-size/4, size)
        plt.yticks([])
        plt.xticks([])
        plt.tight_layout()
    plt.show()


def roc_curve_plots(y_true, model_proba, title='ROC Curve', figsize=(12, 9)):
    ''' Функция построения ROC кривой для моделей из переданнойго словаря

        Принимает:
    '''
    plt.figure(figsize=figsize)
    for name, proba in model_proba.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label='AUC = %0.4f (%s)' % (roc_auc, name))
    plt.title(title)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.grid()
    plt.xlim([-0.01, 1]), plt.ylim([0, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def multiclass_roc_curve_plots(y_true, proba, class_labels=None, title='ROC Curve', 
                               figsize=(12, 9)):
    fpr, tpr, roc_auc = {}, {}, {}
    
    if class_labels is None: class_labels = [0,1,2]
    
    y_true = label_binarize(y_true, classes=[0,1,2])
    n_classes = y_true.shape[1]
    
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=figsize)
    plt.plot(fpr["micro"], tpr["micro"], linestyle=":", lw=4,
        label="AUC = {:0.4f} (micro-average ROC curve)".format(roc_auc["micro"]),
    )
    plt.plot(fpr["macro"], tpr["macro"], ls=":", lw=4,
        label="AUC = {:0.4f} (macro-average ROC curve)".format(roc_auc["macro"]),
    )        
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2,
                 label="AUC = {:0.4f} (Class: {})".format(roc_auc[i], class_labels[i]))
    plt.title(title)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.grid()
    plt.xlim([-0.01, 1]), plt.ylim([0, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
    
def confusion_matricies_plots(confusion_matricies, class_labels=None, figsize=(12,4)):
    import pandas as pd
    _, ax =plt.subplots(1, len(confusion_matricies), figsize=figsize)
    i = 0
    for name, matrix in confusion_matricies.items():
        sns.heatmap(pd.DataFrame(matrix), annot=True, annot_kws={"fontsize":12}, 
                    cmap='Blues', vmin=0, vmax=1, ax=ax[i])
        ax[i].set_title(name, fontsize=14)
        ax[i].set_xlabel('Predictions', fontsize=14)
        ax[i].set_ylabel('True', fontsize=14)
        if class_labels is not None:
            ax[i].set_xticks(np.arange(0.5,len(class_labels)), class_labels, fontsize=12)
            ax[i].set_yticks(np.arange(0.5,len(class_labels)), class_labels, fontsize=12)
        i += 1
    plt.suptitle('Confusion matricies', fontsize=12, y=1.0)
    plt.show()
