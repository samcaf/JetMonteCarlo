import matplotlib.pyplot as plt

# Local imports:
from jetmontecarlo.utils.plot_utils import *
from jetmontecarlo.utils.color_utils import *
from jetmontecarlo.utils.vector_utils import *
from jetmontecarlo.utils.partonshower_utils import *

# Params
radius = 1.
beta = 2

colors = {
          0: OrangeShade4,
          1: PurpleShade1,
          2: RedShade1,
          3: BlueShade1,
          4: GreenShade1
         }

#########################################################
# Visualizer Class:
#########################################################
class Visualizer3D(object):
    # From https://github.com/kinjalh/Physics-Jet-Simulator-3D
    """
    Creates a 3D visualization of a tree representing parton splits.
    The visualization is a plot on 3D axes.
    """

    def __init__(self):
        """
        Initializes a visualizer that plots 3D figures.
        """
        fig = plt.figure(figsize=(10, 10))
        self._axes = fig.add_subplot(111, projection='3d')
        self._axes.set_axis_off()
        self._axes.set_xlabel('X')
        self._axes.set_ylabel('Y')
        self._axes.set_zlabel('Z')

        self._axes.set_xlim(-4,4)
        self._axes.set_ylim(-4,4)
        self._axes.set_zlim(-4,4)

    def draw_partons(self, parton, x_0, y_0, z_0, depth=0):
        """Creates a 3-D plot of the vector of a parton
        with the current node as root. Starts plotting
        the vectors relative to an initial (x, y, z)
        of (x_0, y_0, z_0). Each vector starts where
        its parent vector ends.

        Parameters
        ----------
        parton : Parton
            The root of the tree of vectors to graph
        x_0 : type
            The x coordinate at which the first vector starts.
        y_0 : type
            The y coordinate at which the first vector starts.
        z_0 : type
            The z coordinate at which the first vector starts.

        Returns
        -------
        None
        """
        if parton is not None:
            mom   = parton.momentum.vector
            scale = (1 + 5*np.exp( -parton.momentum.mag()))
            mom   =  mom /(parton.momentum.mag() * scale)

            x_end = x_0 + mom[0]
            y_end = y_0 + mom[1]
            z_end = z_0 + mom[2]

            if parton.isFinalState: color = 'seagreen'
            else: color = colors[depth%4]
            self._axes.plot([x_0, x_end], [y_0, y_end], [z_0, z_end],
                            linewidth=2.0, color=color)
            self._axes.scatter(x_end, y_end, z_end, s=25, alpha=.8,
                                color=color)

            self.draw_partons(parton.daughter1, x_end, y_end, z_end,
                              depth = depth+1)
            self.draw_partons(parton.daughter2, x_end, y_end, z_end,
                              depth = depth+1)

    def draw_cone(self, radius=1):
        """
        Based on https://stackoverflow.com/a/39823124/190597 (astrokeat)
        """
        # vector in direction of axis
        v = [0,1,0]
        n1= [0,0,1]
        n2= [1,0,0]

        # surface ranges over t from 0 to length of axis and 0 to 2*pi
        n = 80
        t = np.linspace(0, 4, n)
        theta = np.linspace(0, 2 * np.pi, n)
        # use meshgrid to make 2d arrays
        t, theta = np.meshgrid(t, theta)
        R = np.linspace(0, radius, n)
        # generate coordinates for surface
        X, Y, Z = [v[i] * t + R * np.sin(theta) * n1[i]
                   + R * np.cos(theta) * n2[i] for i in [0, 1, 2]]
        self._axes.plot_surface(X, Y, Z, color='palegreen', alpha=.2,
                    linewidth=5)

    # matplotlib doesn't work here
    def draw_detector(self):
        def draw_cylinder(radius, color,alpha=1):
            # Cylindrical detector
            x=np.linspace(-radius, radius, 100)
            z=np.linspace(-7, 7, 100)
            Xc, Zc=np.meshgrid(x, z)
            Yc = np.sqrt(radius**2.-Xc**2)

            # Draw parameters
            rstride = 20
            cstride = 10
            self._axes.plot_surface(Xc, -Yc, Zc, alpha=alpha,
                            rstride=rstride, cstride=cstride,
                            linewidth=50, color=color)
        det_rad = 5
        draw_cylinder(det_rad, 'darkcyan')

        inner_rads = [.85,.9,1.05,1.1]
        for rad in inner_rads: draw_cylinder(rad,'goldenrod')

        tracker_rads = [1.9,2.1,2.9,3.1,3.9,4.1]
        for rad in tracker_rads: draw_cylinder(rad, 'skyblue')



    def draw_beampipe(self):
        """Creates a background for the visualization.
        """
        self._axes.scatter(0,0,0, s=300, marker=(8,1,20),
                            color='gold', alpha=.9, zorder=0)
        self._axes.plot(xs=[0, 0], ys=[0, 0], zs=[-6, 6],
                        linewidth=6.0, color='darkred',
                        zorder=0.2)
        self._axes.plot(xs=[0, 0], ys=[0, 0], zs=[-6.1, 6.1],
                        linewidth=2.0, color='mistyrose',
                        zorder=0.2)

#########################################################
# Visualizations:
#########################################################
def visualize_angular_shower(acc='LL'):
    # Proof of concept with just a single shower:
    ang_init = radius**beta / 2.
    momentum = Vector([0,P_T,0])

    jet      = Jet(momentum, radius, partons=None)
    mother   = jet.partons[0]

    angularity_shower(mother, ang_init, beta, 'quark', jet,
                      acc=acc, split_soft=False)
    jet.has_showered = True

    vis = Visualizer3D()
    vis.draw_partons(jet.partons[0], 0, 0, 0)
    vis.draw_beampipe()
    # vis.draw_detector()
    vis.draw_cone()
    plt.show()

#########################################################
# Tests:
#########################################################
if __name__ == "__main__":
    visualize_angular_shower('LL')
