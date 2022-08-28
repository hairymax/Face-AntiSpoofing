# Original code https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
# Author : @zhuyingSeu , Company : Minivision
# Modified by @hairymax

from datetime import datetime


def get_time():
    return (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')


def get_kernel(height, width):
    kernel_size = ((height + 15) // 16, (width + 15) // 16)
    return kernel_size


def parse_model_name(model_name):
    info = model_name.split('_')[0:-1]
    h_input, w_input = info[-1].split('x')
    model_type = model_name.split('.pth')[0].split('_')[-1]

    if info[0] == "org":
        scale = None
    else:
        scale = float(info[0])
    return int(h_input), int(w_input), model_type, scale


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
    import matplotlib.pyplot as plt
    import seaborn as sns
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
