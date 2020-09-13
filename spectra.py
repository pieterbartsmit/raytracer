import numpy
import scipy.interpolate
def raised_cosine(directions, mean_direction, width ):
    lookup_table = [[1., 37.5],
                    [2., 31.5],
                    [3., 27.6],
                    [4., 24.9],
                    [5., 22.9],
                    [6., 21.2],
                    [7., 19.9],
                    [8., 18.8],
                    [9., 17.9],
                    [10., 17.1],
                    [15., 14.2],
                    [20., 12.4],
                    [30., 10.2],
                    [40., 8.9],
                    [50., 8.0],
                    [60., 7.3],
                    [70., 6.8],
                    [80., 6.4],
                    [90., 6.0],
                    [100., 5.7],
                    [200., 4.0],
                    [400., 2.9],
                    [800., 2.0] ]

    lookup_table =  numpy.array(lookup_table).transpose()
    lookup_table = numpy.flip(lookup_table,1)

    if width <= lookup_table[1,0]:
        power = lookup_table[0,0]
    elif width >= lookup_table[1,-1]:
        power = lookup_table[0,-1]
    else:
        interp = scipy.interpolate.interp1d(lookup_table[1, :],
                                            lookup_table[0, :])
        power = interp(width)

    mutual_angle = (mean_direction - directions + 180) % 360 - 180

    D = numpy.where( numpy.abs(mutual_angle) < 90, numpy.cos(mutual_angle*numpy.pi/180.)**power, 0.)
    delta = directions[2]-directions[1]
    return D / (numpy.sum(D) * delta)