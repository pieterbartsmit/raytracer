import numpy as np
from ctypes import c_double,c_float,c_int32,c_int64, POINTER, c_char_p, c_bool
import ctypes
c_double_p = POINTER(c_double)
c_int32_p  = POINTER(c_int32)
c_int64_p  = POINTER(c_int64)
c_float_p  = POINTER(c_float)
c_bool_p   = POINTER(c_bool)


LIBRARY_LOCATION = '/Users/pietersmit/PycharmProjects/raytracer/fortran/raytracing.lib'
lib = ctypes.cdll.LoadLibrary( LIBRARY_LOCATION  )
lib.setnum.restype = None
lib.setnum.argtypes = ( c_double, c_int32, c_int32 )

lib.setcont.restype = None
lib.setcont.argtypes = ( c_double,)

lib.setbndlist.restypes = None
lib.setbndlist.argtypes = ( c_int32_p, c_int32)

lib.setbat.restypes = None
lib.setbat.argtypes = ( c_double_p,c_double_p,c_double_p,c_int32,c_int32)

lib.setdom.restypes = None
lib.setdom.argtypes = ( c_double_p,c_double_p)

lib.calc.restypes = None
#matrix, xp,yp,np, freq, nfreq,angles,nang, nsubrays
lib.calc.argtypes = ( c_double_p ,
                      c_double_p,
                      c_double_p,
                      c_int32,
                      c_double_p,
                      c_int32,
                      c_double_p,
                      c_int32
                      ,c_int32)


class RayTracer:
    #
    def __init__( self , origin_lat_lon, lat_width, lon_width  , dep_lat , dep_lon , dep, bndlist , dcont, workdir='./',
                  save=True,radangles=None,angfreq=None,backwardOnly=True,nsub=100 ):
        #

        #load the external library

        self.nsub = nsub
        #Set the domain
        dom_lon = np.array([origin_lat_lon[1] , origin_lat_lon[1]+lon_width],dtype='float64' )
        dom_lat = np.array([origin_lat_lon[0]  , origin_lat_lon[0] + lat_width],dtype='float64')
        dom_lon_pointer = dom_lon.ctypes.data_as(c_double_p)
        dom_lat_pointer = dom_lat.ctypes.data_as(c_double_p)

        lib.setdom( dom_lon_pointer , dom_lat_pointer)
        
        
        if backwardOnly:
            #
            self.backwardOnly = 1
            #
        else:
            #
            self.backwardOnly = -1
            #        
        
        
        #Set/Load Bathymetry in library
        self.lon  = dep_lon
        self.lat  = dep_lat
        self.dep  = dep
        self.nlon = len(dep_lon)
        self.nlat = len(dep_lat)
        self.n    = self.nlon * self.nlat
        self.workdir =  workdir
        self.save    = save


        d=np.reshape(self.dep,(self.n,) )

        depi = d.astype('float64')
        depi_pointer = depi.ctypes.data_as(c_double_p)

        lati = dep_lat.astype('float64')
        lati_pointer = lati.ctypes.data_as(c_double_p)

        loni = dep_lon.astype('float64')
        loni_pointer = loni.ctypes.data_as(c_double_p)

        nloni = ctypes.c_int(self.nlon)
        nlati = ctypes.c_int(self.nlat)        
        

        lib.setbat( depi_pointer ,
                    loni_pointer,
                    lati_pointer,
                    nloni,
                    nlati)
        #
        self.bndlist = []
        for bnd in bndlist:
            #
            if bnd in ['W','w',1]:
                #
                self.bndlist.append( 1 )
                #
            elif bnd in ['E','e',2]:
                #
                self.bndlist.append( 2 )
                #
            elif bnd in ['S','s',3]:
                #
                self.bndlist.append( 3 )
                #
            elif bnd in ['N','n',4]:
                #
                self.bndlist.append( 4 )                
                #
            #
        #
        bndlisti = np.array(self.bndlist,dtype='int32')
        bndlisti_pointer = bndlisti.ctypes.data_as(c_int32_p)
        nbndlisti = ctypes.c_int( len(self.bndlist) )
        lib.setbndlist( bndlisti_pointer , nbndlisti )

        self.maxstep = 10000
        self.frac    = 1
        lib.setnum( self.frac ,self.maxstep, self.backwardOnly  )


        lib.setcont( dcont )
        self.radangles = radangles
        self.angfreq   = angfreq

    #end __init__

    def getmatrix( self , xp , yp , radangles,nsub, angfreq ):
        #
        # Wrapper to calculate the dependency matrix using the raytracer code
        # NOTE: angles and frequencies are LISTS in terms of RADIANS and
        #       angular frequency!
        #
        # Is there already a workfile...?
        #
        import os

        self.nsub = nsub
        workfile =  self.workdir + 'matrix.npy'
        fileexists = os.path.isfile(workfile)
        if fileexists and self.save:
            #
            # ...if so load the transfermatrix...
            #
            matrix = np.load(workfile)
            #
        else:
            #
            # ... else we calculate the transfermatrix; for each input
            # transform to the appropriate ctype
            #
            nang   = len(radangles)
            matrix = np.zeros( nang * nang * len(angfreq) * len(xp),dtype='float64' )
            matrix_pointer = matrix.ctypes.data_as(c_double_p)


            xp     = np.array(xp,dtype='float64')
            xp_pointer = xp.ctypes.data_as(c_double_p)

            yp     = np.array(yp,dtype='float64')
            yp_pointer = yp.ctypes.data_as(c_double_p)


            nfreq = len(angfreq)
            freq  = np.array(angfreq,dtype='float64')
            freq_pointer = freq.ctypes.data_as(c_double_p)

            angles = np.array(radangles,dtype='float64')
            angles_pointer = angles.ctypes.data_as(c_double_p)
            #
            # ... Call the library ...
            #
            lib.calc.restypes = None
            lib.calc( matrix_pointer,
                      xp_pointer,
                      yp_pointer,
                      len(xp),
                      freq_pointer,
                      nfreq,
                      angles_pointer,
                      nang,
                      nsub)

            #
            # ...Reshape to correct dimensions and return transfer matrix...
            #
            matrix = np.array( matrix ).reshape( ( len(angfreq),len(xp),nang,nang) )
            #
            #
            # ... and save it to a file.
            #
            np.save(workfile,matrix)
            #
        #endif
        #
        return(matrix)        
        #
    #endDef

    def prediction( self , angfreq,radangles ,spec, xp , yp , nsub  ):
        #
        # Predict spectrum at location based on boundary spectrum
        #        
        matrix = self.getmatrix(  xp , yp , radangles , nsub, angfreq )

        ndir=  len(radangles)
        nfreq = len(angfreq)
        #
        # Calculate the local spectrum
        #
        E0 = np.zeros( (len(xp), nfreq, ndir) )
        #
        for jf in range(0, nfreq):
            #
            # For each frequency
            #
            for ip in range( 0 , len(xp) ):
                #
                # and at each point, multply dependency matrix with offshore spectrum
                # to obtain local spectrum
                #
                E0[ip,jf,:]  = np.matmul( matrix[ jf,ip , :, : ] , spec[: , jf ] )
                #
            #end for
            #
        #end for
        #
        # Finally, reorder output...        
        E0 = np.transpose( E0 , [0,2,1] )

        return( E0 )
        #
    #end prediction
#