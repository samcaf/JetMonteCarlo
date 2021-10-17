import numpy as np
import matplotlib as mpl

# ---------------------------------------------------
# pgf formatting:
# ---------------------------------------------------
# See http://bkanuka.com/posts/native-latex-plots/

mpl.use('pgf')

golden_mean = (np.sqrt(5)-1.0)/2.0

fig_width = 246./72.
fig_height = fig_width*golden_mean

params = {
    "pgf.texsystem": "pdflatex",	# Use PdfLaTeX as system
    "text.usetex": True, 			# Render Text with Latex

    "font.family": "serif",

    "text.antialiased" : False,

    "pgf.preamble": [
        # Preamble for PGF,
        # should be the same as the document you are using
        r"\usepackage[utf8x]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[final]{microtype}",
        r"\usepackage[USenglish]{babel}",
        r"\usepackage{newtxtext}",
        r"\usepackage{amsmath}",
        r"\usepackage{newtxmath}",
    ],

    "font.size" : 9,
    "axes.labelsize" : 9,	# Global Font Size
    "xtick.labelsize" : 8,	# XTick Font Size
    "ytick.labelsize" : 8,	# YTick Font Size

    "lines.linewidth" : 0.75,
    "axes.linewidth"  : 0.5,
    "axes.formatter.useoffset" : True,
    "axes.xmargin" : 0,
    "axes.ymargin" : 0,

    "xtick.direction" : 'in',
    "ytick.direction" : 'in',
    "xtick.major.width" : 0.5,
    "xtick.minor.width" : 0.5,
    "ytick.major.width" : 0.5,
    "ytick.minor.width" : 0.5,


    "figure.figsize": [fig_width, fig_height],
    "figure.autolayout": False,
}
mpl.rcParams.update(params)
