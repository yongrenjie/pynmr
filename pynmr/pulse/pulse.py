import numpy as np
from pathlib import Path
import math
from datetime import datetime
PI = math.pi


def polar2cart(amplitudes, phases):
    x = np.multiply(amplitudes, np.cos(phases))
    y = np.multiply(amplitudes, np.sin(phases))
    return x, y


def cart2polar(x, y):
    amplitudes = np.sqrt((x ** 2) + (y ** 2))
    phases = np.arctan2(y, x)
    return amplitudes, phases


def generate(points, duration, bandwidth, smoothing, type, form='polar'):
    # Largely adopted from code in Spinach 2.4.
    # except that the code always returns a 'normalised' pulse with max amplitude of 1.
    duration = float(duration)
    bandwidth = float(bandwidth)
    smoothing = float(smoothing)
    time = np.linspace(-0.5, 0.5, points)

    if type == "wurst":
        if smoothing < 1:
            raise ValueError("For a WURST pulse the smoothing factor / index must be 1 or greater.")
        # Kupce, E.; Freeman, R.
        # J. Magn. Reson. A 1995, 115, 273
        # doi:10.1006/jmra.1995.1179
        A = 1 - (abs(np.sin(PI * time)) ** smoothing)
        # d\phi/dt = \omega = 2\pi\nu
        # \nu = (bandwidth/duration) * t (linear frequency sweep)
        # t = duration * time (because time goes only from -0.5 to +0.5)
        # a simple integration gives the following result (assuming \phi(0) = 0)
        phi = PI * duration * bandwidth * (time ** 2)

    elif type == "chirp":
        if smoothing < 0 or smoothing > 50:
            raise ValueError("For a chirp pulse the smoothing factor must be between 0 and 50.")
        # Bohlen, J.-M.; Bodenhausen, G.
        # J. Magn. Reson. A 1993, 102, 293
        # doi:10.1006/jmra.1993.1107
        # The algorithm for calculating smoothfunc is not the same as in Spinach; it's (marginally) more
        # accurate here. The idea is to convert the following continuous piecewise function to a discrete version: 
        # f(x) = sin(pi*x/(2s))     for x < s
        # f(x) = 1                  for s <= x <= 1-s
        # f(x) = sin(pi*(1-x)/(2s)) for x > 1-s
        # whereas in Spinach, s itself is converted to an integer (by floor()) before evaluating the piecewise
        # function, which causes loss of precision. 
        # The two approaches are equivalent iff (smoothing/100)*points is an integer (i.e. floor() is unnecessary).
        s = smoothing/100
        x = np.linspace(0, 1, points)
        f1 = lambda x : np.sin((PI * x) / (2 * s))
        f2 = lambda x : 1
        f3 = lambda x : np.sin((PI * (1 - x)) / (2 * s))
        A = np.piecewise(x, 
                [x < s, (x >= s) & (x <= 1 - s), x > 1 - s], 
                [f1, f2, f3])
        phi = PI * duration * bandwidth * (time ** 2)

    elif type == 'saltire':
        if smoothing < 0 or smoothing > 50:
            raise ValueError("For a saltire pulse the smoothing factor must be between 0 and 50.")
        # Foroozandeh, M.; Morris, G. A.; Nilsson, M.
        # Chem Eur. J. 2018, 24, 13988
        # doi:10.1002/chem.201800524
        # see 'chirp' above for discussion of piecewise smoothing function
        s = smoothing/100
        x = np.linspace(0, 1, points)
        f1 = lambda x : np.sin((PI * x) / (2 * s))
        f2 = lambda x : 1
        f3 = lambda x : np.sin((PI * (1 - x)) / (2 * s))
        A_chirp = np.piecewise(x,
                [x < s, (x >= s) & (x <= 1 - s), x > 1 - s], 
                [f1, f2, f3])
        # these are phases for a chirp pulse; not correct for a saltire
        phi_chirp = PI * duration * bandwidth * (time ** 2)
        # we correct for it now
        Cx, Cy = polar2cart(A_chirp, phi_chirp)
        # remove the imaginary component, because
        # A\exp(i\phi) + A\exp(-i\phi) = 2A\cos\phi
        Cy = np.zeros(points)
        # transform back to polar coordinates
        A, phi = cart2polar(Cx, Cy)

    else:
        raise ValueError("Unknown type of pulse specified ('{}').".format(type))

    if form == 'polar' or form == None:
        return A, phi
    elif form == 'cart':
        Cx, Cy = polar2cart(A, phi)
        return Cx, Cy
    else:
        raise ValueError("Form '{}' not recognised. Please use either 'polar' or 'cart'.".format(form))


def smoothing(smooth_percentage, points, type):
    # returns a smoothing profile ranging from 0 to 1
    # based on the quarter-sine wave as described in
    # Bohlen, J.-M.; Bodenhausen, G.
    # J. Magn. Reson. A 1993, 102, 293
    # doi:10.1006/jmra.1993.1107
    if type == 'quartersine' or type == None:
        if smooth_percentage < 0 or smooth_percentage > 50:
            raise ValueError("The smoothing factor must be between 0 and 50.")
        s = smooth_percentage/100
        x = np.linspace(0, 1, points)
        f1 = lambda x : np.sin((PI * x) / (2 * s))
        f2 = lambda x : 1
        f3 = lambda x : np.sin((PI * (1 - x)) / (2 * s))
        smoothfunc = np.piecewise(x,
                                  [x < s, (x >= s) & (x <= 1 - s), x > 1 - s],
                                  [f1, f2, f3])
    elif type == 'sinebell':
        if smoothing < 1:
            raise ValueError("The smoothing factor / index must be 1 or greater.")
        # Kupce, E.; Freeman, R.
        # J. Magn. Reson. A 1995, 115, 273
        # doi:10.1006/jmra.1995.1179
        smoothfunc = 1 - (abs(np.sin(PI * time)) ** smoothing)
    return smoothfunc


def writebruk(l1, l2, path, form="polar", title="pynmr.pulse"):
    """ Writes a pulse to Bruker-readable text file, which should be placed in
    /path/to/TopSpin/exp/stan/nmr/lists/wave/user.
    Pulse must be specified as a pair of lists l1 and l2.
    If form is "polar", then l1 and l2 are assumed to be A and phi;
    if form is "cart", then l1 and l2 are the x- and y-coefficients.
    It is assumed that the rf amplitude of the pulse is already contained in l1 and l2,
    i.e. l1 and l2 are not normalised to 1.
    The title of the pulse can be optionally specified. """

    if format == "polar":
        A = l1
        phi = l2
        x, y = polar2cart(A, phi)
    elif format == "cart":
        x = l1
        y = l2
        A, phi = cart2polar(x, y)
    else:
        raise ValueError("write2bruk: form must be specified as either 'polar' or 'cart'.")
    max_amp = max(A)
    A = A * 100 / max_amp # scale so that maximum is 100
    xmax = np.amax(x)
    xmin = np.amin(x)
    ymax = np.amax(y)
    ymin = np.amin(y)
    phi = (phi * 180 / PI) + 180 # convert to degrees in [0, 360]

    t = datetime.now()

    with open(path, 'w') as fid:
        print("##TITLE= {}".format(title), file=fid)
        print("##JCAMP-DX= 5.00 Bruker JCAMP library", file=fid)
        print("##DATA TYPE= Shape Data", file=fid)
        print("##ORIGIN= pynmr.pulse.writebruk()", file=fid)
        print("##OWNER= <Jonathan Yong>", file=fid)
        print("##DATE= " + t.strftime("%Y-%m-%d"), file=fid)
        print("##TIME= " + t.strftime("%H:%M:%S%z"), file=fid)
        print("##$SHAPE_PARAMETERS= Type: optimal control pulse", file=fid)
        print("##MINX= {}".format(xmin), file=fid)
        print("##MAXX= {}".format(xmax), file=fid)
        print("##MINY= {}".format(ymin), file=fid)
        print("##MAXY= {}".format(ymax), file=fid)
        print("##NPOINTS= {}".format(len(A)), file=fid)
        print("##XYPOINTS= (XY..XY)", file=fid)
        print("", file=fid)
        mat = np.array([A, phi])
        np.savetxt(fid, np.transpose(mat), fmt="%.6f")
        print("##END=", file=fid)

    # max_amp is the rf amplitude of the pulse, which needs to be converted to TopSpin's SPW.
    # A simple way to do this is to define the rf amplitude of the shaped pulse as cnstX in TopSpin,
    # then use: spwX = plw1 * (cnstX / (1000000/(4*p1)))^2 
    return max_amp
