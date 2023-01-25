from jetmontecarlo.utils.pgfplot_utils import *

import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, LogLocator, FixedFormatter, FixedLocator)

savefig = False

# ------------------------------------
# Test code from David Newsom:
# ------------------------------------
ax = plt.axes()

x = np.linspace(2,12,1000)
dx = [2, 4, 6, 8, 10, 12]

r1 = 0.5
r2 = 0.75
r3 = 0.9

p1 = -r1/np.sqrt(1-np.power(r1,2))
p2 = -r2/np.sqrt(1-np.power(r2,2))
p3 = -r3/np.sqrt(1-np.power(r3,2))

nVals1 = [2.82277, 8.83813, 25.7943, 70.3032, 164.836, 298.218]
nVals2 = [5.62308, 37.7346, 193.113, 467.49, 586.237, 608.264]
nVals3 = [13.8919, 192.722, 593.294, 666.001, 670.313, 670.545]

theory1 = np.sqrt(r1/2)*np.sqrt(np.power((1+r1)/(1-r1),x) - 1)/(r1+(np.power((1+r1)/(1-r1),x/2)-(r1*x-1))/1000)
theory2 = np.sqrt(r2/2)*np.sqrt(np.power((1+r2)/(1-r2),x) - 1)/(r2+(np.power((1+r2)/(1-r2),x/2)-(r2*x-1))/1000)
theory3 = np.sqrt(r3/2)*np.sqrt(np.power((1+r3)/(1-r3),x) - 1)/(r3+(np.power((1+r3)/(1-r3),x/2)-(r3*x-1))/1000)

xur1 = -1/r1*np.sqrt(x/2)*(p1/np.sin(np.pi/x))*(np.sqrt(np.sin(np.pi/x)**2+p1**2)-p1)/(1-2*x/1000/np.sin(np.pi/x)**2*p1*np.sqrt(np.sin(np.pi/x)**2+p1**2))
xur2 = -1/r2*np.sqrt(x/2)*(p2/np.sin(np.pi/x))*(np.sqrt(np.sin(np.pi/x)**2+p2**2)-p2)/(1-2*x/1000/np.sin(np.pi/x)**2*p2*np.sqrt(np.sin(np.pi/x)**2+p2**2))
xur3 = -1/r3*np.sqrt(x/2)*(p3/np.sin(np.pi/x))*(np.sqrt(np.sin(np.pi/x)**2+p3**2)-p3)/(1-2*x/1000/np.sin(np.pi/x)**2*p3*np.sqrt(np.sin(np.pi/x)**2+p3**2))


plt.semilogy(x,theory1,'r', alpha=0.5, zorder=5)
plt.semilogy(x,theory2,'k', alpha=0.5, zorder=5)
plt.semilogy(x,theory3,'b', alpha=0.5, zorder=5)
plt.semilogy(x, xur1, 'r:', alpha=0.5, zorder=1)
plt.semilogy(x, xur2, 'k:', alpha=0.5, zorder=1)
plt.semilogy(x, xur3, 'b:', alpha=0.5, zorder=1)
plt.plot(dx, nVals1, 'rs', markersize=4, zorder=10)
plt.plot(dx, nVals2, 'ko', markersize=4, zorder=10)
plt.plot(dx, nVals3, 'b^', markersize=4, zorder=10)


plt.ylim(1,1000)
plt.xlim(1.75,12.25)
plt.xlabel(r'Number of Membranes')
plt.ylabel(r'$g_c/g_1$', labelpad=2.5)
plt.xticks([2,4,6,8,10,12])


ax.yaxis.set_major_locator(FixedLocator([10,100]))
ax.yaxis.set_major_formatter(FixedFormatter([r'$10$',r'$100']))
ax.yaxis.set_minor_locator(FixedLocator([2,3,4,5,6,7,8,9,20,30,40,50,60,70,80,90,200,300,400,500,600,700,800,900]))
ax.yaxis.set_minor_formatter(FixedFormatter(['','','',r'$5$','','','','','','','',r'$50$','','','','','','','',r'$500$','','','','']))


plt.tight_layout(pad=0.05)

if savefig: plt.savefig('samplepgfplot.pdf')
