import colorsys
import matplotlib.colors as mpc

# ---------------------------------------------------
# List of single colors:
# ---------------------------------------------------
MyRed = '#B72525'
MyGreen = '#228922'
MyBlue = '#4A4AF3'
MyTurquoise = '#26A5B4'
MyGrey = '#8F8F8F'
MyGrey2 = '#4B4B4B'
MyRed2 = '#F10E09'
MyBlue2 = '#3257FE'
MyLightGrey = '#DFDFDF'
MyLightRed = '#EABDBD'
MyLightBlue = '#ADADFF'
MyLightGreen = '#BDEEBD'

BlueShade1 = '#1E3B6E'
BlueFace1 = '#5E99DC'

BlueShade2 = '#3B6D9D'
BlueFace2 = '#9CCAEB'

BlueShade3 = '#778AAB'
BlueFace3 = '#D0D2D0'

BlueShade4 = '#A1C1F7'
BlueFace4 = '#D9DFED'

BlueShade5 = '#96b9bb'
BlueShade6 = '#1e3476'

GreenShade1 = '#5F6C33'
GreenFace1 = '#60AA25'

GreenShade2 = '#569225'
GreenFace2 = '#BFD666'

GreenFace3 = '#E4F7D1'

RedShade1 = '#842525'
RedFace1 = '#D24F4F'

RedShade2 = '#C43737'
RedFace2 = '#ED8A8C'

RedShade3 = '#DA7B7B'
RedFace3 = '#ECBDBD'

RedFace4 = '#FFDADA'

GreyShade = '#232630'
GreyFace = '#B1B2B3'

OrangeShade1 = '#ED8E0C'
OrangeFace1 = '#FFBE49'

OrangeShade2 = '#EF9D0A'
OrangeFace2 = '#FFCE75'

OrangeShade3 = '#FEB03C'
OrangeFace3 = '#FFE2BC'

OrangeShade4 = '#bd6e00'
OrangeFace4 = '#efac2e'

PurpleShade1 = "#6900E2"
PurpleFace1 = "#BF88FF"

PurpleShade2 = "#9E49F3"
PurpleFace2 = "#D4AEFB"

def adjust_lightness(color, amount=0.5):
    """
    Adjusts the lightness of the given color by multiplying
    (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> adjust_lightness('g', 0.3)
    >> adjust_lightness('#F034A3', 0.6)
    >> adjust_lightness((.3,.55,.1), 0.5)

    From https://stackoverflow.com/a/49601444
    """
    try:
        c = mpc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mpc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

# ---------------------------------------------------
# Color Dictionaries:
# ---------------------------------------------------

# Colors I found aesthetic for use in comparison plots
compcolors = {
    (-1, 'dark'): 'black',
    (0, 'dark'): OrangeShade4,
    (1, 'dark'): PurpleShade1,
    (2, 'dark'): RedShade1,
    (3, 'dark'): MyBlue,
    (4, 'dark'): GreenShade1,
    (-1, 'medium'): 'dimgrey',
    (-1, 'light'): 'lightgrey',
    (0, 'light'): OrangeShade3,
    (1, 'light'): PurpleShade2,
    (2, 'light'): RedShade3,
    (3, 'light'): MyLightBlue,
    (4, 'light'): GreenShade2
    }

lightnessmed = [1.1, 1.3, 1.4, 1.2, 1.2]
col_medium = {(i, 'medium'):adjust_lightness(compcolors[(i, 'dark')],
                                 amount=lightnessmed[i])
              for i in range(5)}

compcolors.update(col_medium)

# Different instantiations of the 'light' colors above,
# for use in, for example, errorbar plots where the errorbars
# are of a lighter color
lightness1 = [1.3, 1.3, 1.4, 1.4, 1.4]
col_light1 = {i:adjust_lightness(compcolors[(i, 'dark')],
                                 amount=lightness1[i])
              for i in range(4)}
lightness2 = [2.2, 1.8, 2.4, 2.5, 2.5]
col_light2 = {i:adjust_lightness(compcolors[(i, 'dark')],
                                 amount=lightness2[i])
              for i in range(4)}
