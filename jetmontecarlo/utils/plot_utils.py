from datetime import date
from math import atan2, degrees

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import container
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerErrorbar

from matplotlib.path import Path
from matplotlib.patches import PathPatch

# Local imports
from jetmontecarlo.utils.color_utils import *

# ---------------------------------------------------
# Formatting:
# ---------------------------------------------------
_small_size = 10
_medium_size = 12
_bigger_size = 14
_large_size = 16

plt.rc('font', size=_medium_size)         # controls default text sizes
plt.rc('axes', titlesize=_bigger_size)    # fontsize of the axes title
plt.rc('axes', labelsize=_bigger_size)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=_small_size)    # fontsize of the tick labels
plt.rc('ytick', labelsize=_small_size)    # fontsize of the tick labels
plt.rc('legend', fontsize=_medium_size)   # legend fontsize
plt.rc('figure', titlesize=_large_size)   # fontsize of the figure title

# ---------------------------------------------------
# Styles:
# ---------------------------------------------------
# Line plot style
style_solid = {'ls':'-', 'lw':2}
style_dashed = {'ls':'--', 'lw':2}

# Scatter plot style
style_scatter = {'s':1, 'alpha':.9}

# Errorbar plot styles
style_yerr = {'xerr':0, 'markersize':2.5, 'fmt':'s',
              'elinewidth':2.3, 'capsize':0, 'zorder':1}
style_yerr_ps = {'xerr':0, 'markersize':4, 'fmt':'o',
              'elinewidth':2.3, 'capsize':0, 'zorder':.5}
# MIT open data style (P. Komiske)
modstyle = {'lw':2, 'capsize':2, 'capthick':1.5, 'markersize':2,
            'linestyle':'None', 'zorder':1}
modstyle_ps = {'lw':2, 'capsize':2, 'capthick':1.5, 'markersize':5,
               'fmt':'o', 'linestyle':'None', 'zorder':.5}

#########################################################
# Plot creation
#########################################################
# ---------------------------------------------------
# Basic figure type:
# ---------------------------------------------------
def aestheticfig(xlabel='x', ylabel=r'Probability Density',
                 title=None, showdate=True,
                 xlim=(0, 1), ylim=(0, 1), ylim_ratio=(0.5, 2.),
                 ratio_plot=True, ylabel_ratio='Ratio',
                 labeltext='JetMC'):
    """Creates a figure and associated axes. Can be used to
    produce a figure with a subplot which is, for example,
    associated with a ratio.

    Parameters
    ----------
    xlabel : str
        xlabel of the plot.
    ylabel : str
        ylabel of the plot.
    title : str
        title of the plot.
    showdate : bool
        If True, adds a date to the upper right of the plot.
    xlim : tuple
        The x limits of the plot.
    ylim : tuple
        The y limits of the plot.
    ylim_ratio : tuple
        The y limits of the ratio subplot.
    ratio_plot : bool
        Determines whether there is an additional subplot
        for ratio plotting.
    ylabel_ratio : str
        ylabel of the ratio subplot, if it exists.

    Returns
    -------
    Figure, axes.Axes
        The figure and axes/subplots specified by the
        above parameters.
    """
    # aesthetic options
    # fig_width = 5.
    # golden_mean = (np.sqrt(5)-1.0)/2.0
    # fig_height = fig_width/golden_mean
    fig_width = 6.4
    fig_height = 4.8
    figsize = (fig_width, fig_height)

    gridspec_kw = {'height_ratios': (3.5, 1) if ratio_plot else (1,),
                   'hspace': 0.0}

    # get subplots
    nsubplots = 2 if ratio_plot else 1
    fig, axes = plt.subplots(nsubplots, gridspec_kw=gridspec_kw,
                             figsize=figsize)
    if nsubplots == 1:
        axes = [axes]

    # axes limits
    for ax in axes:
        ax.set_xlim(*xlim)
    axes[0].set_ylim(*ylim)
    if ratio_plot:
        axes[1].set_ylim(*ylim_ratio)
        axes[1].set_yscale('log')

    # axes labels
    axes[-1].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel, labelpad=5)
    if ratio_plot:
        axes[1].set_ylabel(ylabel_ratio, labelpad=-10)

    # tick settings
    for ax in axes:
        ax.minorticks_on()
        ax.tick_params(top=True, right=True, bottom=True,
                       left=True, direction='in', which='both')

    if ratio_plot:
        axes[0].tick_params(labelbottom=False)
        axes[1].tick_params(axis='y')

    # Extra plot information
    pad = .01

    if showdate:
        # Including date
        axes[0].text(
            x=1,
            y=1.005+pad,
            s=date.today().strftime("%m/%d/%y"),
            transform=axes[0].transAxes,
            ha="right",
            va="bottom",
            fontsize=_medium_size * 0.95,
            fontweight="normal"
        )

    if labeltext is not None:
        # Extra primary label
        axes[0].text(
            x=-0.1,
            y=1.005+pad,
            s=labeltext,
            transform=axes[0].transAxes,
            ha="left",
            va="bottom",
            fontsize=_medium_size * 1.5,
            fontweight="bold",
            fontname="DIN Condensed"
        )

    if title is not None:
        # Main title
        axes[0].text(
            x=.12,
            y=1.005+pad,
            s=title,
            transform=axes[0].transAxes,
            ha="left",
            va="bottom",
            fontsize=_medium_size * 1.5,
            fontstyle="italic",
            fontname="Arial"
        )

    plt.tight_layout()

    return fig, axes

# ---------------------------------------------------
# Putting text on figures:
# ---------------------------------------------------

def set_figtext(fig, text, loc, rightjustify=False, color='black'):
    """Puts text of a nice style on figures."""
    if rightjustify:
        ha = 'right'
    else:
        ha = 'left'

    t = fig.text(*loc, text, linespacing=1.5, ha=ha, color=color)
    t.set_bbox(dict(facecolor='white', alpha=0.9,
                    edgecolor='lightgrey',
                    boxstyle="round,pad=0.35"))


def stamp(left_x, top_y, ax=None, delta_y=0.075, textops_update=None, **kwargs):
    """Function from MODplot library to add stamp to figures."""
     # handle defualt axis
    if ax is None:
        ax = plt.gca()

    # text options
    textops = {'horizontalalignment': 'left',
               'verticalalignment': 'center',
               'fontsize': 8.5,
               'transform': ax.transAxes}
    if isinstance(textops_update, dict):
        textops.update(textops_update)

    # add text line by line
    for i in range(len(kwargs)):
        y = top_y - i*delta_y
        t = kwargs.get('line_' + str(i))
        if t is not None:
            ax.text(left_x, y, t, **textops)

#########################################################
# Error Bands
#########################################################
# From https://matplotlib.org/stable/gallery/lines_bars_and_markers/curve_error_band.html
def draw_error_band(ax, x, y, err, normal=False, **kwargs):
    if normal:
        # Calculate normals via centered finite differences (except the first point
        # which uses a forward difference and the last point which uses a backward
        # difference).
        dx = np.concatenate([[x[1] - x[0]], x[2:] - x[:-2], [x[-1] - x[-2]]])
        dy = np.concatenate([[y[1] - y[0]], y[2:] - y[:-2], [y[-1] - y[-2]]])
        l = np.hypot(dx, dy)
        nx = dy / l
        ny = -dx / l

        # end points of errors
        xp = x + nx * err
        yp = y + ny * err
        xn = x - nx * err
        yn = y - ny * err
    else:
        xn, xp = x, x
        yn, yp = y - err, y + err

    vertices = np.block([[xp, xn[::-1]],
                         [yp, yn[::-1]]]).T
    codes = np.full(len(vertices), Path.LINETO)
    codes[0] = codes[len(xp)] = Path.MOVETO
    path = Path(vertices, codes)
    ax.add_patch(PathPatch(path, **kwargs))


#########################################################
# Labelling lines
#########################################################
# From https://stackoverflow.com/a/39402483

def labelLine(line, x, label=None, align=False, **kwargs):
    """Labels a line with the corresponding
    line2d label data.
    """
    ax = line.axes
    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if (x < xdata[0]) or (x > xdata[-1]):
        print('x label location is outside data range!')
        return

    #Find corresponding y co-ordinate and angle of the line
    ip = 1
    for i, xdata_i in enumerate(xdata):
        if x < xdata_i:
            ip = i
            break

    y = (ydata[ip-1] +
         (ydata[ip]-ydata[ip-1])
         *(x-xdata[ip-1])
         /(xdata[ip]-xdata[ip-1]))

    if not label:
        label = line.get_label()

    if align:
        #Compute the slope
        dx = xdata[ip] - xdata[ip-1]
        dy = ydata[ip] - ydata[ip-1]
        ang = degrees(atan2(dy, dx))

        #Transform to screen co-ordinates
        pt = np.array([x, y]).reshape((1, 2))
        trans_angle = ax.transData.transform_angles(
            np.array((ang,)), pt)[0]

    else:
        trans_angle = 0

    #Set a bunch of keyword arguments
    if 'color' not in kwargs:
        kwargs['color'] = line.get_color()

    if ('horizontalalignment' not in kwargs) and ('ha' not in kwargs):
        kwargs['ha'] = 'center'

    if ('verticalalignment' not in kwargs) and ('va' not in kwargs):
        kwargs['va'] = 'center'

    if 'backgroundcolor' not in kwargs:
        kwargs['backgroundcolor'] = ax.get_facecolor()

    if 'clip_on' not in kwargs:
        kwargs['clip_on'] = True

    if 'zorder' not in kwargs:
        kwargs['zorder'] = 2.5

    t = ax.text(x, y, label, rotation=trans_angle, **kwargs)
    t.set_bbox(dict(facecolor='white', alpha=0.9,
                    edgecolor='lightgrey',
                    boxstyle="round,pad=0.15"))

def labelLines(lines, align=False, xvals=None,
               spacing=None, **kwargs):
    """Labels a set of lines with the
    corresponding line2D label data.
    """
    ax = lines[0].axes
    labLines = []
    labels = []

    #Take only the lines which have labels other than the default ones
    for line in lines:
        label = line.get_label()
        if "_line" not in label:
            labLines.append(line)
            labels.append(label)

    if xvals is None:
        xmin, xmax = ax.get_xlim()
        if spacing is None or spacing == 'lin':
            xvals = np.linspace(xmin, xmax, len(labLines)+2)[1:-1]
        elif spacing == 'log':
            xvals = np.logspace(np.log10(xmin), np.log10(xmax),
                                len(labLines)+2)[1:-1]

    for line, x, label in zip(labLines, xvals, labels):
        labelLine(line, x, label, align, **kwargs)

#########################################################
# Legend utilities
#########################################################
def legend_yerr(axes, loc='best'):
    """Makes a legend for the ax object 'axes' at the location 'loc',
    but only includes the y error errorbars in the legend icons.

    Parameters
    ----------
    axes : axes.Axes
        The axes on which we want to create a legend with only
        y error bars
    loc : str
        The location of the legend.
    """
    handles, labels = axes.get_legend_handles_labels()

    new_handles = []

    for _, h in enumerate(handles):
        #only need to edit the errorbar legend entries
        if isinstance(h, container.ErrorbarContainer):
            new_handles.append(
                container.ErrorbarContainer(
                    h.lines, has_xerr=False, has_yerr=True
                    )
                )
        else:
            new_handles.append(h)

    axes.legend(new_handles, labels, loc=loc)

def legend_darklight(axes, darklabel='Monte Carlo',
                     lightlabel='Analytic',
                     errtype=None, twosigma=False,
                     extralabel=None):
    """Makes a legend for the ax object 'axes' at the location
    'loc', which indicates dark solid lines as 'MC' (or darklabel)
    and light dotted lines as 'Analytic' (or lightlabel).

    Parameters
    ----------
    axes : axes.Axes
        Description of parameter `axes`.
    darklabel : str
        Label of the dark objects in the legend.
    lightlabel : str
        Label of the light objects in the legend.
    errtype : str
        Specifies the plot type of the `dark` data,
        and thus the corresponding legend icon.
        None (default): legend for line plot.
        yerr: legend for a plot with ecolor
        modstyle: legend for a modstyle errorbar plot
    twosigma : bool
        Determines whether to include lighter, two sigma
        error bars in the legend.

    Returns
    -------
    type
        Description of returned object.

    """
    if errtype is None:
        custom_lines = [Line2D([0], [0], **style_solid,
                               color=compcolors[(-1, 'dark')]),
                        Line2D([0], [0], **style_dashed,
                               color=compcolors[(-1, 'light')])]

        axes.legend(custom_lines, [darklabel, lightlabel])

    elif errtype == 'yerr':
        _, xmax = axes.get_xlim()
        axes.errorbar(xmax*50., 0, yerr=1., **style_yerr,
                      color='black', ecolor='gray', label=darklabel)

        if twosigma:
            axes.errorbar(xmax*50., 0, yerr=2., **style_yerr,
                          color='black', ecolor='lightgray',
                          label=darklabel)
        if extralabel is not None:
            axes.errorbar(xmax*50., 0, yerr=1., **style_yerr_ps,
                          color=compcolors[(-1, 'medium')],
                          ecolor=compcolors[(-1, 'light')],
                          label=extralabel)
            if twosigma:
                axes.errorbar(xmax*50., 0, yerr=2., **style_yerr_ps,
                              color=compcolors[(-1, 'medium')],
                              ecolor=compcolors[(-1, 'light')],
                              label=extralabel)

        handles, _ = axes.get_legend_handles_labels()

        if twosigma:
            l = 0
            # Setting up containers for both errorbars
            if extralabel is not None:
                twosig_extra = container.ErrorbarContainer(
                    handles[-1].lines, has_xerr=False,
                    has_yerr=True)
                onesig_extra = container.ErrorbarContainer(
                    handles[-2].lines, has_xerr=False,
                    has_yerr=True)
                l = -2
            twosig = container.ErrorbarContainer(
                handles[l-1].lines, has_xerr=False,
                has_yerr=True)
            # Setting up containers for both errorbars
            onesig = container.ErrorbarContainer(
                handles[l-2].lines, has_xerr=False,
                has_yerr=True)
            if extralabel is not None:
                custom_handles = [(twosig, onesig),
                                  (twosig_extra, onesig_extra)]
            else:
                custom_handles = [(twosig, onesig)]
        else:
            l = 0
            # Setting up containers for both errorbars
            if extralabel is not None:
                onesig_extra = container.ErrorbarContainer(
                    handles[-1].lines, has_xerr=False,
                    has_yerr=True)
                l=-1
            onesig = container.ErrorbarContainer(
                handles[l-1].lines, has_xerr=False,
                has_yerr=True)
            if extralabel is not None:
                custom_handles = [onesig, onesig_extra]
            else:
                custom_handles = [onesig]

        custom_handles.append(
            Line2D([0], [0], **style_dashed, color=compcolors[(-1, 'light')])
            )

        if twosigma:
            if extralabel is not None:
                axes.legend(custom_handles, [darklabel, extralabel, lightlabel],
                            handler_map={
                                onesig: HandlerErrorbar(xerr_size=0.37),
                                twosig: HandlerErrorbar(xerr_size=0.65),
                                onesig_extra: HandlerErrorbar(xerr_size=0.37),
                                twosig_extra: HandlerErrorbar(xerr_size=0.65)}
                            )
            else:
                axes.legend(custom_handles, [darklabel, lightlabel],
                            handler_map={
                                onesig: HandlerErrorbar(xerr_size=0.37),
                                twosig: HandlerErrorbar(xerr_size=0.65)}
                            )
        else:
            if extralabel is not None:
                axes.legend(custom_handles, [darklabel, extralabel, lightlabel])
            else:
                axes.legend(custom_handles, [darklabel, lightlabel])

    elif errtype == 'modstyle':
        _, xmax = axes.get_xlim()
        axes.errorbar(xmax*50., 0, yerr=1., **modstyle,
                       color=compcolors[(-1, 'dark')],
                       label=darklabel)
        if extralabel is not None:
            axes.errorbar(xmax*50., 0, yerr=1., **modstyle_ps,
                          color=compcolors[(-1, 'medium')],
                          label=extralabel)

        handles, _ = axes.get_legend_handles_labels()

        # Setting up containers for errorbars
        l = 0
        if extralabel is not None:
            onesig_extra = container.ErrorbarContainer(
                handles[-1].lines, has_xerr=True,
                has_yerr=True)
            l=-1
        onesig = container.ErrorbarContainer(handles[l-1].lines,
                                             has_xerr=True,
                                             has_yerr=True)
        if extralabel is not None:
            custom_handles = [onesig, onesig_extra]
        else:
            custom_handles = [onesig]

        custom_handles.append(
            Line2D([0], [0], **style_dashed, color=compcolors[(-1, 'light')])
            )

        if extralabel is not None:
            axes.legend(custom_handles, [darklabel, extralabel, lightlabel],
                        prop={'size': 15}, loc='upper left')
        else:
            axes.legend(custom_handles, [darklabel, lightlabel])
