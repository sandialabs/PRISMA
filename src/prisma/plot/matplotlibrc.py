import matplotlib.pyplot as plt


def set_rc_journal():
    '''
    Figure style (journal): 4:3 panels, inward ticks on all sides,
    1.0 pt lines, 12 pt text
    '''
    PANEL_W, PANEL_H = 4.0, 3.0  # inches, 4:3 aspect per panel
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 12,
        "lines.linewidth": 1.0,
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "figure.figsize": (PANEL_W, PANEL_H),
    })


def set_rc_journal_const_wh():
    '''
    Figure style (journal): default panels, inward ticks on all sides,
    1.0 pt lines, 12 pt text
    '''
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 12,
        "lines.linewidth": 1.0,
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "figure.figsize": (4, 3),
        "figure.constrained_layout.use": True,
        "figure.dpi": 100
    })
