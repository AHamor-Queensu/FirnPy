#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors  ## import LinearSegmentedColormap
from matplotlib import cm
from mpl_toolkits import mplot3d
import seaborn as sns  ## package that deals with heat maps
import re  ## Regex needed for line parsing
import csv
import struct
import os  ## needed for file size
import string  ## needed to filter non-printable characters from comment
import datetime
import copy
from scipy.signal import filtfilt, butter, tukey, cheby1, bessel, firwin, lfilter, wiener
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

#
# Notes:
#   ToDo
#
#   1. new pinch points from parabola() should replace pinchPoints
#
#   2. make filenames (.DT1 and .HD) and, perhaps, a file of parameters passable
#
#   3. figure out memory management in Python
#
#

#
# Parse the HD file
#    
#    The file is a list of Ascii strings that must be parsed, and the useful data
#      put in a dictionary (thanks GPRpy for that idea).
#    
def parseHD(text_file):
    """ Parse the .HD ASCII header file.

    Parameters:
        text_file: String containing the path, filename and extension of the .HD file

    Function:
        Parse the ASCII strings in the HD file for pertinent strings which are stored in a ‘hd’ dictionary

    Return Values:
        hd - dictionary of needed or interesting values
        twtt - an np array of two way travel times for the Trace
    """
    print("parseHD", text_file)
    
    txt_fp = open(text_file, "r")
    file_size = os.path.getsize(text_file)

    Lines = txt_fp.readlines()
 
    count = 0
    # Elemental file parser
    # line.strip() Strips the newline character
    datestamp = []
    numTraces = 0
    numPoints = 0
    frequency = 200  # default frequency in Megahurtz
    seperation = 0   # antenna separation in meters
    timeZero = 0
    totalTimeWindow = 0
    maxElevation = 0
    minElevation = 0
    notFloat = 0     # used as an action taken when ValueError
    hd = {}          # dictionary of .HD information
    
    for line in Lines:
        count += 1

        if (len(line) < 6): # Ignore the "1234"
            continue
        string = line.strip()
        if (count == 3): # System
            system = string
            hd["system"] = string
        elif (count == 5): # Datestamp
            datestamp = string
            hd["date"] = string
        elif ("NUMBER OF TRACES" in line):
            var = re.match(r'NUMBER OF TRACES   = (.*)', string)
            hd["numTraces"] = int(var.group(1))
        elif ("NUMBER OF PTS/TRC" in line):
            var = re.match(r'NUMBER OF PTS/TRC  = (.*)', string)
            numPoints = int(var.group(1))
            hd["numPoints"] = numPoints
        elif ("TIMEZERO AT POINT" in line):
            var = re.match(r'TIMEZERO AT POINT  = (.*)', string)
            hd["timeZero"] = float(var.group(1))
        elif ("TOTAL TIME WINDOW" in line): # time in nanoseconds
            var = re.match(r'TOTAL TIME WINDOW  = (.*)', string)
            hd["timeWindow"] = float(var.group(1))
        elif ("STARTING POSITION" in line):
            var = re.match(r'STARTING POSITION  = (.*)', string)
            hd["startPos"] = float(var.group(1))
        elif ("FINAL POSITION" in line):
            var = re.match(r'FINAL POSITION     = (.*)', string)
            hd["finalPos"] = float(var.group(1))
        elif ("STEP SIZE USED" in line):
            var = re.match(r'STEP SIZE USED     = (.*)', string)
            hd["stepSize"] = float(var.group(1))
        elif ("POSITION UNITS" in line):
            var = re.match(r'POSITION UNITS     = (.*)', string)
            hd["posUnits"] = str(var.group(1))
        elif ("NOMINAL FREQUENCY" in line):
            var = re.match(r'NOMINAL FREQUENCY  = (.*)', string)
            hd["freq"] = float(var.group(1))
        elif ("ANTENNA SEPARATION" in line):
            var = re.match(r'ANTENNA SEPARATION = (.*)', string)
            hd["antSeperation"] = float(var.group(1))

    txt_fp.close() # Because there are such thing as manners
                                   
    # Transform feet to meters (nods to GPRpy for this hint)
    if hd['posUnits'] == 'ft':
        hd["Start_pos"] = hd["Start_pos"]*0.3048
        hd["Final_pos"] = hd["Final_pos"]*0.3048
        hd["Step_size"] = hd["Step_size"]*0.3048
        hd["antSeperation"] = hd["antSeperation"]*0.3048
        hd['Pos_units'] = 'm'
    
    sec_per_samp = hd["timeWindow"]/hd["numPoints"]
    time_to_timeZeroPoint = int(hd["timeZero"] * sec_per_samp)


    twtt = np.linspace(0,hd["timeWindow"],hd["numPoints"])-time_to_timeZeroPoint
    #
    # velocity in meters per nanoseconds m/ns
    #   speed of light in vaccuum is 3.0 x10^8 m/s or .3 m/ns
    #   speed of light in ice is 2.29 x 10^8 m/s or .229 m/ns CRC Handbook 48th Editioin and College Physics 6th Edition
    #   speed of light in firn is rounded to .2 m/ns - rule of thumb, no source given
    #
    velocity = 0.20  # m/ns - meters per nanosecond
    depthVelocity = twtt * velocity/2.0 

    
    # length and area per sample
    #
    #  velocity in firn (nanometers/sec)/2 [two way travel time] * (total time / total number of points)
    depthPerSample = 0.2/2 * hd["timeWindow"]/hd["numPoints"]
    hd["depthPerSample"] = depthPerSample
    areaPerSample = round(hd["stepSize"] * depthPerSample,6) 
    hd["areaPerSample"] = areaPerSample

    print("")
    print("hd dictionary:")
    for key, value in hd.items():
        print("  ",key,": ", value)
    print("")
    
    return hd, twtt

#
# Parse the DTn file
#    Break trace records apart into header and data
#       Header consists of 25 4 byte floats followed by a 28 character string
#       Data is an array of two byte (signed short) values, one per data point (pointPerTrace)
#            which is copied into a list - strength - which is appended to a list of lists
#            labelled (pedantically) SA (for Strength Array)
#    Currently the data is recorded, and any Comments are printed
#    The last negative value greater than lastNegativeThreshold (-100) is recorded for later processing
#    The structure of the header can be found in appendix A of the Ekko Project Manual
#
#     Also note,  the file position must be calculated as Python does not have an EOF
#
def parseDTn(binary_file):
    """ Parse the .DT1 binary data file.

    Parameters:
        binary_file: String containing the path, filename and extension of the .DT1 file

    Function:
        Parse the ASCII and the binary fields in the DT1 file for the strength array. Other values are parsed but do not appear to be needed.

    Return Values:
        SA - strength array of integer values in a list of lists
    """
    print("parseDTn: filename", binary_file)
    numPos = numNeg = numZero = 0
    maxPos = xMaxPos = yMaxPos = 0  # position of maximun positive value
    maxNeg = xMaxNeg = yMaxNeg = 0  # position of maximum negative value
    lowSigNeg = xLowSigNeg = yLowSigNeg = 0 # position of lowest significant (-100) negative
    sigNegThreshold = -100
    gpsPoint = [] # list pf gps x, y, z coordinates
    gps = [] # list of gps points
    gps_x_0 = 0.0 # first gps point
    gps_y_0 = 0.0
    gps_z_0 = 0.0
    gps_x_l = 0.0 # last gps point
    gps_y_l = 0.0
    gps_z_l = 0.0
    tod_0 = 0   # TimeOfDay first
    tod_l = 0   # TimeOfDay last
    position_0 = 0.0 # fist and last position
    position_l = 0.0

    SA = []  # the currently empty Strength Array (list of list)
   

    b_fp = open(binary_file, "rb")
    file_size = os.path.getsize(binary_file)

    while True:  ## Go forever
        ## struct and unpack are used to extract binary data from .DT#
        ## call the set of 4 as a float not an int. Returned as a tuple
        trace = struct.unpack('f', b_fp.read(4))  # The trace number (as a tuple)
        position = struct.unpack('f', b_fp.read(4))  # The position (as a tuple) in meters
        if position_0 == 0.0:
            position_0 = position[0]
        position_l = position[0]
        ppT = struct.unpack('f', b_fp.read(4))  # The ppt - points per trace (as a tuple)
        pointPerTrace = int(ppT[0])  # Convert points per trace to an integer
        topographic_data = struct.unpack('f', b_fp.read(4))  # Topographic data (as a tuple)
        struct.unpack('f', b_fp.read(4))  # not used

        bytesPerPoint = struct.unpack('f', b_fp.read(4))  # bytes/point (always 2 for Rev 3 firmware)
        if (int(bytesPerPoint[0]) != 2):  ## If the data is from an older version, the code will not work
            print(" Aborting: bytesPerPoint should be 2 - software cannot cope with other that firmware ver3 data",
                  int(bytesPerPoint[0]))
            exit(0)

        timeWindow = struct.unpack('f', b_fp.read(4))  # time window (as a tuple) The maximum time selected for viewing,

        numStacks = struct.unpack('f', b_fp.read(4))  # number of stacks (as a tuple), number of repeated measurements averaged to get resulting measurements
        gps_x = struct.unpack('d', b_fp.read(8))  # float can be either 4 or 8 bytes
        #gpsPoint.append(gps_x)
        gps_y = struct.unpack('d', b_fp.read(8))
        #gpsPoint.append(gps_y)
        gps_z = struct.unpack('d', b_fp.read(8))
        #gpsPoint.append(gps_z)
        #gpsPoint = list((gps_x, gps_y, gps_z))
        gpsPoint =[gps_x, gps_y, gps_z]
        gps.append(gpsPoint)
        if (int(gps_x[0]) != 0):  # if the GPS data is available, print it
            if (gps_x_0 == 0.0): # store the first gps point
                gps_x_0 = gps_x[0]
                gps_y_0 = gps_y[0]
                gps_z_0 = gps_z[0]
            # store the next so that the last is stored
            gps_x_l = gps_x[0]
            gps_y_l = gps_y[0]
            gps_z_l = gps_z[0]

        rx_x = struct.unpack('f', b_fp.read(4))  # receiver x,y,x (as tuples)
        rx_y = struct.unpack('f', b_fp.read(4))
        rx_z = struct.unpack('f', b_fp.read(4))
        if (int(rx_x[0]) != 0):
            print(" Rx x y z ", int(rx_x[0]), int(rx_y[0]), int(rx_z[0]))

        tx_x = struct.unpack('f', b_fp.read(4))  # transmitter x,y,z (as tuples)
        tx_y = struct.unpack('f', b_fp.read(4))
        tx_z = struct.unpack('f', b_fp.read(4))
        if (int(tx_x[0]) != 0):
            print(" Tx x y z ", int(tx_x[0]), int(tx_y[0]), int(tx_z[0]))

        t0 = struct.unpack('f', b_fp.read(4))  # timezero adjustment  (as a tuple)
        if (int(t0[0]) != 0):  # if non-zero, print timezero adjustment
            print(" t0 ", int(t0[0]))
        zero_flag = struct.unpack('f', b_fp.read(4))  # Zero flag (as a tuple), 0 = data okay, 1=zero data
        if (int(zero_flag[0]) != 0):  # if non-zero, print zero_flag
            print(" t0 ", int(t0[0]))
        struct.unpack('f', b_fp.read(4))  # not used
        tod = struct.unpack('f', b_fp.read(4))  # time of day (as a tuple) in seconds past midnight
        if tod_0 == 0:
            tod_0 = int(tod[0])
        else:
            tod_l = int(tod[0])
        comment_flag = struct.unpack('f', b_fp.read(4))  # comment flag
        comment = (b_fp.read(28))  # all comments seen have had unprintable ASCII

        printable = set(string.printable)

        if (comment_flag[0] == 1):
            ''.join(filter(lambda x: x in printable, comment))
            print(" comment", printable)

        # Data pointPerTrace 2byte ints
        # fmt = \"%dH\" % (pointPerTrace) # format 'dH' unsigned short
        fmt = "%dh" % (pointPerTrace)  # format 'dh' signed short
        strength = []  # strength of reflection at time and distance (Note - should this be initialized outside loop?)
        strength = list(struct.unpack(fmt, b_fp.read(pointPerTrace * 2)))

        ## This is a way at looking for the divide between the weak and strong signals for the purpose of applying two
        ## different gain functions

        for i in range(0, len(strength)):
            if ((strength[i] < sigNegThreshold) and (yLowSigNeg < i)): 
                lowSigNeg = strength[i]
                yLowSigNeg = i
                xLowSigNeg = int(trace[0])
            if (strength[i] > maxPos):
                maxPos = strength[i]
                yMaxPos = i
                xMaxPos = int(trace[0])
            if (strength[i] < maxNeg):
                maxNeg = strength[i]
                yMaxNeg = i
                xMaxNeg = int(trace[0])
            if (strength[i] > 0):
                numPos += 1
            elif (strength[i] < 0):
                numNeg += 1
            else:
                numZero += 1

        SA.append(strength)

        pos = b_fp.tell()
        if pos == file_size:  # EOF
            break

    b_fp.close()  # Because we are civilized

    print(" trace ", trace[0], " position ", position[0], " time window " , timeWindow[0] , " time of day ", int(tod[0]), " pointPerTrace ", pointPerTrace)
    print(" first position {0:.3f}".format(position_0), " last position {0:.3f}".format(position_l), "delta position: {0:.3f}".format(position_l - position_0))
    print(" time of day: start ", tod_0, " stop ", tod_l, " delta ", tod_l - tod_0, " minutes ", int((tod_l - tod_0)/60), " seconds ", (tod_l - tod_0)%60 )
    print("  velocity: {0:.2f}".format((position_l - position_0)/(tod_l - tod_0)), "meters/second")
    print(" first gps {0:.7f}".format(gps_y_0)," {0:.7f}".format(gps_x_0), " {0:.7f}".format(gps_z_0))
    print(" last  gps {0:.7f}".format(gps_y_l)," {0:.7f}".format(gps_x_l), " {0:.7f}".format(gps_z_l))

    #
    #  Perform a Sanity check on the file just read
    #
    if (len(SA) < 3):  # IF the file has less than three columns - what can be analysed?
        print(" The file contains two or less columns - processing shall terminate")
        exit(0)
    if (len(strength) < 3):  # IF the file has less than three rows - what can be analysed?
        print(" The file contains two or less rows - processing shall terminate")
        exit(0)
    NotZero = 0  ## Check if the given data set is an empty list of lists
    for i in range(0, len(SA)):
        for j in range(0, len(strength)):
            if SA[i][j] != 0:
                NotZero = 1
                break
        if NotZero == 1:
            break
    if NotZero == 0:
        print(" Aborting: No data has been found in .DT# file - in ", len(SA), " by ", len(strength))
        exit(0)
    #
    # Some notable stats
    #
    print("  numPos", numPos, "numNeg", numNeg, "numZero", numZero)
    if (lowSigNeg):
        print(" Lowest significant negative: x={0:4d} y={1:4d} value={2:6d} threshold= {3:4d}".format(xLowSigNeg, yLowSigNeg, lowSigNeg, sigNegThreshold))
    if (maxPos):
        print(" Maximum positive:            x={0:4d} y={1:4d} value={2:6d}".format(xMaxPos, yMaxPos, maxPos))
    if (maxNeg):
        print(" Maximum negative:            x={0:4d} y={1:4d} value={2:6d}".format(xMaxNeg, yMaxNeg, maxNeg))
    print("")        
    return SA


def window(Trace, upperX, upperY, lowerX, lowerY, Data=False):
    """ Extract a section of the Trace for analysis

    Parameters:
        Trace: strength array as a list of list
        upperX,upperY,lowerX,lowerY: coordinates of the section
        Data: print 'window' section in 10x10 squares

    Function:
        Extract a section of the Strength Array for analysis of the duplicate Trace comparison

    Return Values:
        window - the section of the array
    """
    print("window: upperX ",upperX," upperY ",upperY," lowerX ",lowerX," lowerY ",lowerY)
    #
    ## extract a 'window' section of Trace
    #
    #
    ## Sanity Check
    if (lowerX < 0 or lowerY < 0):
        if (lowerX < 0):
            print(" lower X input as ", lowerX, " reset to zero")
            lowerX = 0
        else:
            print(" lower Y input as ", lowerY, " reset to zero")
            lowerY = 0
    if (upperX > len(Trace) or upperY > len(Trace[0])):
        if (upperX > len(Trace)):
            print(" upper X input as ", upperX, " which exceeds Trace length ", len(Trace))
            upperX = len(Trace)
        else:
            print(" upper Y input as ", upperY, " which exceeds Trace depth ", len(Trace[0]))
            upperY = len(Trace[0])
    #
    ## Check if co-ordinates make sense
    #
    if (lowerX < upperX):
        print(" Reversing ", lowerX, upperX)
        temp = lowerX
        lowerX = upperX
        upperX = temp
    if (lowerY < upperY):
        print(" Reversing ", lowerY, upperY)
        temp = lowerY
        lowerY = upperY
        upperY = temp
    #
    ## extract 'window' section
    #
    lenX = lowerX - upperX 
    lenY = lowerY - upperY
    print(" lenX ", lenX, " lenY " , lenY)
    window = np.zeros([lenX, lenY], int)
    idx = 0
    for x in range(upperX, lowerX):
        window[idx] = Trace[x][upperY:lowerY]
        idx += 1
    
    #
    ## print 'window' section in 10x10 squares
    #
    if (Data == True):
        square = 10
        print(" start at ", upperX, upperY)
        for i in range(0, lenX, square):
                for j in range(0, lenY, square):
                    print(" {0:4d},{1:4d}: ".format(i + upperX, j + upperY))
                    for k in range(0,square):
                        for l in range(0,square):
                            print(" {0:4d} ".format(window[i+k,j+l]), end='')
                        print("")
                    print("")

    return window

def graphData(Trace, Title, cmap_type = 'seismic', Gap=200, someData=True):
    """ Graph the Trace using mathplotlib.

    Parameters:
        Trace: strength array as a list of list
        Title: the title of the graph
        cmap_type: colour map
        Gap: the timeZero point
        someData: print a section of the Trace for visual inspection

    Function:
        Extract a section of the strength array for analysis of the duplicate Trace comparison

    Return Values:
        none
    """
    print("graphData:", Title)
    #
    ## graph the Trace data
    #
    if (someData == True):
        print(" graphData Trace 2000,220:240", Trace[2000][220:240])
        print(" graphData Trace 2000,1600:1620", Trace[2000][1600:1620])
        print("")
         
    yHeat = np.linspace(1, len(Trace[0]), len(Trace[0]))
    xHeat = np.linspace(1, len(Trace), len(Trace))
    XHeat, YHeat = np.meshgrid(xHeat, yHeat)

    ZHeat = Trace

    newgrid = [[x[i] for x in ZHeat] for i in range(len(ZHeat[0]))]
    fig, ax = plt.subplots()

    #color_map = plt.cm.get_cmap('seismic')
    #reversed_color_map = color_map.reversed()

    c = ax.pcolormesh(XHeat, YHeat, newgrid, cmap=cmap_type)

    plt.grid(b=True, which='major', color='#000000', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#000000', linestyle='-', alpha=0.2)
    plt.title(Title)
    plt.xlabel('Trace')
    plt.ylabel('Time (ns)')
    ax.set_ylim(len(Trace[0]), Gap)
    fig.set_size_inches(36, 36)
    return

def graphDataColour(Trace, Title, cmap_type='seismic', window=0, depth=False, save=False):
    """Creates a colour map of a GPR data set. Default colour is seismic, alternatives
    can be found at https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
    
    DOES NOT WORK WITH ELEVATION CORRECTION"""
    
    print("graphDataColour:", Title)
    
    # Convert the x and y co-ords into a way to be mapped onto a 3d mesh
    yHeat = np.linspace(0, len(Trace[0])-1, len(Trace[0]))
    xHeat = np.linspace(0, len(Trace)-1, len(Trace))
    
    # x-axis correction using step size
    stepSizeUsed = hd["stepSize"] #Manual input from .HD file --> make parser to read or figure out how it gets this number ????
    print("HELP",stepSizeUsed,hd["timeZero"],hd["timeWindow"])
    xHeat = stepSizeUsed*xHeat
    # y-axis correction using 
    sampleRate = 0.4
    timeOffset = hd["timeZero"]#Manual input from .HD file ????
    timeScale = (hd["timeWindow"]-(timeOffset * sampleRate))/(len(Trace[0])-timeOffset) # ???? THIS IS WRONG ????
    yHeat = yHeat-timeOffset
    print("Values:", len(Trace[0]), timeScale)
    
    # Set display to either position vs depth or position vs TWT
    if depth == False:
        yHeat = timeScale*yHeat
    if depth == True:
        velocity = 0.2 # ???? speed of light in ice is 0.23 m/ns so why choose 0.2 m/ns ????
        yHeat = velocity/2 * yHeat * timeScale
    
    XHeat, YHeat = np.meshgrid(xHeat, yHeat)
    ZHeat = Trace
    display = [[x[i] for x in ZHeat] for i in range(len(ZHeat[0]))] #Don't remember what stackEx this was grabbed from
    fig, ax = plt.subplots()
    
    #Create the colour map
    c = ax.pcolormesh(XHeat, YHeat, display, cmap=cmap_type)
    
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(24)
    
    #Make the plot - CURRENTLY NOT EDITABLE ????
    #   Grid
    ax.xaxis.set_major_locator(MultipleLocator(200))
    ax.xaxis.set_minor_locator(AutoMinorLocator(10))
    ax.tick_params(which='both', width=2)
    ax.tick_params(which='major', length=7)
    ax.tick_params(which='minor', length=4)
    if depth == False:
        ax.yaxis.set_major_locator(MultipleLocator(100))
        ax.yaxis.set_minor_locator(AutoMinorLocator(10))
    if depth == True:
        ax.yaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(10))
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.5)
    #ax.grid(which='minor', linestyle='', linewidth='0.5', color='black')
    
    if window != 0:
        if window[0] == 'x':
            plt.xlim(window[1], window[2])
        if window[0] == 'y':
            plt.ylim(window[3], window[4])
        if window[0] == 'both':
            plt.xlim(window[1], window[2])
            plt.ylim(window[3], window[4])

    # Invert y axis
    plt.gca().invert_yaxis()
    #   Titles
    plt.title(Title, fontsize=36)
    plt.xlabel("Position (m)", fontsize=30) # ????
    if depth == False:
        plt.ylabel("Time (ns)", fontsize=30)
    if depth != False:
        plt.ylabel("Depth (m)")
    #   Set dimensions and save if enabled
    fig.set_size_inches(39, 20)
    if save == True:
        plt.savefig(Title, bbox_inches='tight')
    return

def removeAverage(Trace, threshold, numTraces=1):
    """ Remove the horizontal average of the Trace.

    Parameters:
        Trace: strength array as a list of list
        threshold: the maximum allowable value (positive)
        numTraces: the number of horizontal lines to average

    Function:
        Create a new strength array where the horizontal average has been removed, convert all values to positive, and limit the maximum value. This is a very destructive process and is only used for visual representations.

    Return Values:
        newTrace - the trace list of list with horizontal average removed and points limited to a maximum value
    """
 
    print("removeAverage: remove average from each line, threshold", threshold, "numTraces",numTraces)
    #
    # Remove the average from each point on line, assumes absolute values
    # Note: a value for numTraces flags that the newTrace shold remain signed
    #

    newTrace = copy.deepcopy(Trace)
    Signed = False
    if (numTraces == 0):
        numTraces = 1
        Signed = True
    
    xRange = len(newTrace)
    yRange = len(newTrace[0])
    removed = 0
    maxValue = 0
    offset = 0
    lineAvgs = []
    
    if ((numTraces < 1) or (numTraces > (int)(yRange/3))): # Sanity check
        print(" numTraces", numTraces," reset to 1")
        numTraces = 1
        offset = 0
    else:
        # make numTraces odd - so that it has equal number of lines before and after
        if (numTraces%2 == 0):
            numTraces += 1
            print(" numTraces increased to ",numTraces)
        average = 0
        offset = int(numTraces/2)
        for y in range(offset):
            for x in range(xRange-1):
                if (newTrace[x][y] < 0):
                    average -= newTrace[x][y]
                    if Signed == False:
                        newTrace[x][y] *= -1 # make absolute
                else:
                    average += newTrace[x][y]
            lineAvgs.append(average)

    for y in range(yRange-1):
        average = 0
        if (y+offset < yRange):
            yoff = y + offset
            for x in range(xRange-1):
                if (newTrace[x][yoff] < 0):
                    average -= newTrace[x][yoff]
                    if Signed == False:
                        newTrace[x][yoff] *= -1 # make absolute
                else:
                    average += newTrace[x][yoff]
            if (len(lineAvgs) < numTraces):
                lineAvgs.append(average)
            else:
                lineAvgs.pop(0)
                lineAvgs.append(average)
            
        average = round(sum(lineAvgs)/(xRange*numTraces)) + 1
        for x in range(xRange-1):
            if (abs(newTrace[x][y]) < average):
                newTrace[x][y] = 0
                removed += 1
            else:
                if Signed == False:
                    newTrace[x][y] -= average
                    if (newTrace[x][y] > threshold):
                        newTrace[x][y] = threshold
                else:
                    if (newTrace[x][y] > 0):
                        newTrace[x][y] -= average
                        if (newTrace[x][y] > threshold):
                            newTrace[x][y] = threshold
                    else:
                        newTrace[x][y] += average
                        if (newTrace[x][y] < -1*threshold):
                            newTrace[x][y] = -1*threshold
                
    print(" average removed: ", removed)
    
    return newTrace

def detectDuplicates(winTrace, threshold=0):
    """ Check for 80% similarities within a thin threshold.

    Parameters:
        winTrace: a sample strength array returned from window()
        threshold: the maximum allowable variation (positive)

    Function:
        If, perchance, a snowmobile dragging a GPR unit stops to retrieve a notebook and the unit is left running creating line artifacts, list them.

    Return Values:
        removal_list - the list Trace columns that have been detected ass duplicates - see removeDuplicates for their fate
    """    
    print("detectDuplicates: threshold ", threshold)
    #
    # Look for duplicate columns and produce a list
    #   Can be deleted via removeDuplicates() 
    #
    xRange = len(winTrace)
    yRange = len(winTrace[0])
    limit = (int)(yRange*2/3)
    histogram = [0] * xRange
    
    for y in range(yRange-1):
        for x in range(xRange-1):
            if (abs(winTrace[x+1][y] - winTrace[x][y]) <= threshold):
                histogram[x] += 1
    total = 0
    last = 0
    removal_list = []
    for x in range(xRange):
        if (histogram[x] > limit):
            total += 1
            if ((last + 1) == x):
                removal_list.append(x)
            last = x

    if (len(removal_list) < 40):
        print(" Removal list:", removal_list)
    
    return removal_list

def removeDuplicates(Trace, removal_list):
    """ Remove a list of duplicates from the Trace.

    Parameters:
        Trace: strength array as a list of list
        removal_list: a list of lines that are duplicates 

    Function:
        Delete duplicate columns from Trace

    Return Values:
        window - the section of the array
    """
  
    print("removeDuplicates:",removal_list)
    #
    # Remove duplicate columns found in removal_list
    #
    previousLength = len(Trace)
    for x in reversed(removal_list):
        Trace.pop(x)
    print(" length now",len(Trace),"from",previousLength)
 
    return Trace


def verticalBandPass(hd, data, low, high, order=5, filttype='butter', cheb_rp=5,fir_window='hamming'):
    """ Remove high and low frequencies from Trace.

    Parameters:
        hd: dictionary of values from HD file
        data: an np array transposed from Trace
        low: the frequency in MHz below which signals are removed
        high: the frequency in MHz above which signals are removed
        Order: the order of the filter - see
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
        filttype: the filter type - options are
            butter, tukey, cheby1, bessel, firwin, lfilter, wiener
            Cheb_rp: the Chebyshev type I filter level
            fir_window: the finite impulse response filter (fir) window type

    Function:
        Wanting a bandpass filter and being way over my head, the kind folks at ImpDar and GPRpy had sources on line to guide these efforts in using scipy functionality. Many thanks. 

    Return Values:
        window - the section of the array
    """
    print("verticalBandPass: high",high,"low",low,"filter",filttype)
    """
    Derived from ImpDAR and GPRPy  whose code is found online (thankfully) and the style of code formatting
    from GPRPy would be used if this code is re-written when I'm a better python programmer. The code is
    kept intact in the hope that further experimetation might provide more insight.

    """

    # first determine the cut-off corner frequencies - expressed as a
    # fraction of the Nyquist frequency (half the sample freq).
    # Note: all of this is in Hz
    #  sample_freq = 1.0 / self.dt  # dt=time/sample (seconds)
    dt = (hd["timeWindow"]*1.0e-9)/float(hd["numPoints"])
    sample_freq = 1.0 / dt # dt=time/sample (seconds)


    # Calculate the Nyquist frequency
    nyquist_freq = 0.5 * sample_freq

    low_corner_freq = low * 1.0e6
    high_corner_freq = high * 1.0e6

    
    corner_freq = np.zeros((2,))
    corner_freq[0] = low_corner_freq / nyquist_freq
    corner_freq[1] = high_corner_freq / nyquist_freq


    # provide feedback to the user
    print('  Bandpassing from {:4.1f} to {:4.1f} MHz...'.format(low, high))

    # FIR operates a little differently, and cheb has rp arg,
    # so we need to do each case separately
    if filttype.lower() in ['butter', 'butterworth']:
        b, a = butter(order, corner_freq, 'bandpass')
        data = filtfilt(b, a, data, axis=0).astype(data.dtype)
    elif filttype.lower() in ['cheb', 'chebyshev']:
        b, a = cheby1(order, cheb_rp, corner_freq, 'bandpass')
        data = filtfilt(b, a, data, axis=0).astype(data.dtype)
    elif filttype.lower() == 'bessel':
        b, a = bessel(order, corner_freq, 'bandpass')
        data = filtfilt(b, a, data, axis=0).astype(data.dtype)
    elif filttype.lower() == 'fir':
        taps = firwin(order + 1, corner_freq, pass_zero=False)
        data[:-order, :] = lfilter(taps, 1.0, data, axis=0).astype(data.dtype)[order:, :]
    else:
        raise ValueError('Filter type {:s} is not recognized'.format(filttype))
    return data



def lookForBand(Trace, Bands, yp=800, ystop=0, depth=5, width=7, threshold=1200):
    """ Traverse the Trace from column bottom to top looking for
        densities indicating a band (of firn or bedrock).

    Parameters:
        Trace: strength array as a list of list
        Bands: an zeroed np.array for bands
        yp,ystop: the upper and lower y limits to search
        depth,width: the dimension of the rectangle to sample strengths
        threshold: the strength found in a rectangle to indicate a band

    Function:
        Search the Trace in rectangle sections looking for strengths above threshold. Positive results are added to the Bands array

    Return Values:
        none
    """
    print("lookForBand: yp=",yp,"ystop=",ystop,"depth=",depth,"width=",width,"threshold=",threshold)
    #
    # Look for the top band(s) i.e. the differentiateable
    #
    # Starting at yp and moving upwards, look for densities of backscatter that fill
    # a box of greater than DEPTH and of WIDTH traces. If the energy density/backscatter
    # is greater than THRESHOLD, the line is recorded in Bands, a three dimensional np
    # array of band where a BAND is structured:
    #
    # Bands[nTraces][nBands][nData=7]
    #   [0] - y co-ordinate (maxY)
    #   [1] - depth of band in strengths(i.e. pixels)
    #   [2] - band density
    #   [3] - x co-ordinate (Trace#)
    #   [4] - already rendered/used (0=False/1=True)
    #   [5] - moving average depth
    #   [6] - moving average midpoint
    #
    bandsShape = Bands.shape # 0 is maxTrace, 1 is maxBands, 2 is maxValues
    BandsIdx = 0
    bandSegFound= 0
    half = (int)(width/2)
    threshold *= half # adjust threshold to width of sample size -- CHECK!
    threshChunk = (int)(threshold/width)
    for x in range(len(Trace)):    
        nBandsIdx = 0
        y = yp
        while (y > ystop):
            sumStr = 0
            threshTotal = 0
            for i in range(x-half, x+half):
                if (i < 0):
                    continue
                if (i >= len(Trace)):
                    break
                threshTotal += threshChunk
                sumStr += abs(Trace[i][y])
            if (sumStr < threshTotal):  # if the sumStr of the width points is less than threshold
                y -= 1               # keep looking
                continue
            
            # The first width is over threshold, now check that the next DEPTH-1 widths
            # are also over threshold
            yStart = y
            density = sumStr
            y -= 1
            while (y > ystop):
                sumStr= 0
                for i in range(x-half, x+half):
                    if (i < 0):
                        continue
                    if (i >= len(Trace)):
                        break
                    sumStr += abs(Trace[i][y])
                # Check depth to see if a band has been accumulated over 'depth'
                if (sumStr < threshTotal):
                    if ((yStart - y) >= depth): # if 'depth' acheived, record the band
                        Bands[BandsIdx][nBandsIdx][0] = yStart
                        Bands[BandsIdx][nBandsIdx][1] = (yStart - y)
                        Bands[BandsIdx][nBandsIdx][2] = density
                        Bands[BandsIdx][nBandsIdx][3] = x
                        Bands[BandsIdx][nBandsIdx][4] = False
                        Bands[BandsIdx][nBandsIdx][5] = 0 # RFU
                        Bands[BandsIdx][nBandsIdx][6] = 0 # RFU
                        nBandsIdx += 1
                        bandSegFound += 1
                        if (nBandsIdx >= bandsShape[1]):
                            print(" lookForBand: nBandsIdx", nBandsIdx, "maximum reached - aborting. bandsShape",bandsShape)
                            y = ystop
                            break
                    y -= 1
                    break
                else:
                    density += sumStr
                    y -= 1
        BandsIdx += 1
        if (nBandsIdx >= bandsShape[0]):
            print(" lookForBand: nBandsIdx", nBandsIdx, "maximum reached - aborting")
            break
    print(" lookForBand: banSegFound", bandSegFound)
    return

def lookForLines(Bands, nBands = 61, maxNumLines = 21, minLineLength = 40):
    """ Search Bands array for contiguous line fragments.

    Parameters:
        Bands: np array of bands
        nBands: maximum number of bands
        maxNumLines: the maximum number of lines to return
        minLineLength: the minimum length of a line 

    Function:
        Algorithmically search for lines in the Bands array by calling findLine()

    Return Values:
        allLines - list of lists of lines found
    """
    print("lookForLines: nBands",nBands,"maxNumLines",maxNumLines,"minLineLength",minLineLength)
    #
    # lookForLines converts the np array of Bands to a list of lines
    #    the routine calls findLine until almost all points
    #     are used or the maximum numbers of lines has been exceeded.
    #   findLine returns a line - a list of (y,depth,density,x,usedFlag,0,0)
    #    which is appended to allLines - a list of lines
    #
    # returns allLines - a list of lines
    #
    
    allLines = []
    validLines = 0
    bandsShape = Bands.shape
    # Count the number of band points recoreded in Bands
    countPoints = 0
    usedPoints = 0
    gappedPoints = 0
    minY = bandsShape[0]
    maxY = 0
    myList = []
    for i in range(bandsShape[0]):
        for j in range(bandsShape[1]):
            if (Bands[i][j][0] != 0):
                countPoints += 1

    xIdx = 0
    for look in range(bandsShape[0]-minLineLength):
        line = []
        myList = findLine(Bands, nBands, line, xIdx, usedPoints, gappedPoints)
        if myList[1] == -1 and myList[2] == -1:  # Reached the end of Bands without finding line
            break
        if gappedPoints == -1: # Empty Bands column
            print("   found empty Band column at idx",xIdx+1)
        else:
            gappedPoints += myList[2]
        aLine = myList[0]
        usedPoints += myList[1]
        if (myList[3] < minY):
            minY = myList[3]
        if (myList[4] > maxY):
            maxY = myList[4]
        xIdx = myList[5]
        if usedPoints == 0 and gappedPoints == 0:
            break
        if (len(aLine) > minLineLength):
            validLines += 1
            allLines.append(aLine)

        if (validLines > maxNumLines):
            print(" Maximum lines reached:",validLines)
            break
        if ((countPoints - usedPoints) < minLineLength): # if there are less points left than minLineLen then stop
            break

    return allLines

def findLine(Bands, nBands, line, xIdx, pointsFound, pointsGapped):
    """ Search for the next available line in Bands.

    Parameters:
        Bands: np array of bands
        nBands: current number of bands
        line: the maximum number of lines to return
        xIdx: the x starting point
        pointsFound: current pointsFound
        pointsGapped: current pointsGapped

    Function:
        Searches for the next line in the Bands array.

    Return Values:
        (line,pointsFound,pointsGapped,minY,maxY,xIdx)
    """
    #print("findLine: line",line,"xIdx",xIdx," and Bands type", type(Bands))
    #
    # Look for the first unUsed point in Bands
    # Find most yDelta adjacent point in next column
    # If not found, try to bridge a single column gap
    #    synthesize gap point via averages
    #
    # returns (line, pointsFound, pointsGapped, minY, maxY)
    #
    # ToDo:
    #    1. Stats gathering- total: area, density  average: depth, density  max/min: depth, density
    #    2. Negative spaces - the ice areas? - not implementd
    #    3. Code optimization
    #    4. expand line for moving 5 averages of depth and midPoint
    #             -currently tacked on as 0,0
    #
    
    #
    # Look for the next free point
    #
    startNBand = 0
    Found = False
    for i in range(xIdx, len(Bands)-2): # 2 columns border for gap bridging
        for j in range(0, nBands):
            startNBand = j
            if (Bands[i][j][4] == True):
                continue
            if (Bands[i][j][0] == 0):
             
                xIdx += 1
                break
            else: # There is data that has not been used, start here
                Found = True
                break
        if (Found == True):
            break
    if (Found != True):
        print("  reached end of Bands",xIdx,startNBand)
        return ([0,0,0,xIdx,0,0,0,0],-1,-1,0,0,xIdx)
   
    x = Bands[xIdx][startNBand][3]
    y = Bands[xIdx][startNBand][0]
    depth = Bands[xIdx][startNBand][1]
    density = Bands[xIdx][startNBand][2]
    minY = maxY = y
    

    line.append([Bands[xIdx][startNBand][0],Bands[xIdx][startNBand][1],Bands[xIdx][startNBand][2],Bands[xIdx][startNBand][3],Bands[xIdx][startNBand][4],0,0])
    
    numMovingAvgPoints = 96                                        # empirically chose - may lead to fragility
    avg = [Bands[xIdx][startNBand][0]] * numMovingAvgPoints    # moving average pre-load
    movingAvg=(int)(sum(avg)/len(avg))
 
    
    # Signify the start of the line as Found - otherwise, if the line length is 
    #   below limit, this will prevent an infinite loop
    Bands[xIdx][startNBand][4] = True
    
    BridgeSkip = 0
    lastUsedBand = 0  # used in bridging gaps
    DepthThreshold = 80  # only allow matches within a certain length - else try gap
    BIG = len(Bands)  # arbitrary large number since len(Trace[0]) not available
    pointsFound = 0
    pointsGapped = 0

    # 
    # The method for finding a line given a starting point is to find the closest point
    #  one column to the right and look for the closest y delta but if there is no point 
    # within the threshold deltaY then check one column over and bridge the gap
    #
    for i in range(xIdx+1, len(Bands)):
        if (BridgeSkip > 0): # Tired and headache so not turning for into while loop as is proper
            BridgeSkip -= 1
            continue
        closest = BIG
        closestIdx = BIG
        for j in range(nBands):
            if (Bands[i][j][0] == 0): # check if data exists - y zeroed indicates no data in column
                pointsFound = 1
                break

            if (Bands[i][j][4] == True):
                continue
            deltaY = abs(movingAvg - Bands[i][j][0])            
            if (deltaY < closest):
                if (deltaY < depth+DepthThreshold): # add wriggle room of depth
                    closest = deltaY
                    closestIdx = j
                    closestPoint = Bands[i][j][0]
                else:
                    closest = BIG # preserve option for gap bridging
        #
        # if the closest point is within threshols then simply append it to the line
        # else check for a band gap bridging solution
        #
        if (closest < BIG): # valid near point found
            lastUsedBand = closestIdx
            if ((Bands[i][closestIdx][3] < peakPinchPoint[0]) or (Bands[i][closestIdx][3] > peakPinchPoint[1])):
                avg.pop(0)
                avg.append(Bands[i][closestIdx][0])
                movingAvg=(int)(sum(avg)/len(avg))
          
            Point = closestPoint            
            Bands[i][closestIdx][4] = True
            Bands[i][closestIdx][5] = sampleTrace[Bands[i][closestIdx][3]][Bands[i][closestIdx][0]]
            line.append([Bands[i][closestIdx][0],Bands[i][closestIdx][1],Bands[i][closestIdx][2],Bands[i][closestIdx][3],Bands[i][closestIdx][4],Bands[i][closestIdx][5],0,0])
            if (Bands[i][closestIdx][0] < minY):
                minY = Bands[i][closestIdx][0]
            if (Bands[i][closestIdx][0] > maxY):
                maxY = Bands[i][closestIdx][0]
            pointsFound += 1
        else: # attempt band gap bridging solution
            bg = i+1
            if (bg >= len(Bands)):
                break
            Bridged = False
            closest = BIG
            closestIdx = BIG
            for a in range(len(Bands[bg])):
                if (Bands[bg][a][4] == True): # point already used
                    continue
                if (Bands[bg][a][0] == 0): # no more points in column
                    break
                deltaY = abs(movingAvg - Bands[bg][a][0])            
                if (deltaY < closest):

                    if (deltaY < depth+DepthThreshold): # add wriggle room of depth
                        closest = deltaY
                        closestIdx = a
       
            if (closest < BIG): # found a free and close point
                Bridged = True
                BridgeSkip = 1
                Bands[bg][closestIdx][4] = True
                # This is synthesis so it could be better
                avgHeight = (int)((Bands[i-1][lastUsedBand][0]+Bands[bg][closestIdx][0])/2)
                avgWidth = (int)((Bands[i-1][lastUsedBand][1]+Bands[bg][closestIdx][1])/2)
                avgDensity = (int)((Bands[i-1][lastUsedBand][2]+Bands[bg][closestIdx][2])/2)
                avgX = (int)((Bands[i-1][lastUsedBand][3]+Bands[bg][closestIdx][3])/2)
                line.append([avgHeight,avgWidth,avgDensity,Bands[bg][closestIdx][3]-1,True,0,0])
                line.append([Bands[bg][closestIdx][0],Bands[bg][closestIdx][1],Bands[bg][closestIdx][2],Bands[bg][closestIdx][3],Bands[bg][closestIdx][4],0,0])
                if (Bands[bg][closestIdx][0] < minY):
                    minY = Bands[bg][closestIdx][0]
                if (Bands[bg][closestIdx][0] > maxY):
                    maxY = Bands[bg][closestIdx][0]
                lastUsedBand = closestIdx
                pointsFound += 1
                pointsGapped += 1
 
            if (Bridged == False):
                break
    return (line,pointsFound,pointsGapped,minY,maxY,xIdx)

def cleave(Bands, Bands_shape, threshold, cleaveList, cleaveThreshold = 25, peakPinchPoint = [2080,2140], peakPinchThreshold = 10):
    """ Separate Lines which have merged.

    Parameters:
        Bands: np array of bands
        Bands_shape: the dimensions of Bands
        Threshold: threshold value for a line
        cleaveList: returned for auditing
        cleaveThreshold: threshold value for splitting a line
        peakPinchPoint: tuple for the start and stop of a pinch point
        peakPinchThreshold: minimum band width

    Function:
        Lines from Bands merge an this algorithm tries to chisel them apart

    Return Values:
        cleaveList - used for auditing
    """

    print("cleave: cleaveThreshold", cleaveThreshold, "peakPinchPoint", peakPinchPoint, "peakPinchThreshold", peakPinchThreshold)
    # 
    # Given the assumption that there should be more than one band in the Bands np array,
    # look for cases where there was only one band which, if it is of a certain depth, could
    # be a merge of two bands. A list of cleaved points is kept to check if this is an 
    # appropriate measure.
    #     cleaveList -> list of tuples (x, y, width)
    #
    # return cleaveList
    #
    bandHist = [0] * Bands_shape[1]
    cleavedCount = 0
    count = 0
    #
    for i in range(Bands_shape[0]):
        if (Bands[i][0][3] == 0): #if the x co-ordinate is zero, then there is no data
            continue
        depthA = 0
        if (Bands[i][1][0] == 0 and Bands[i][0][0] > 0):
            # Check if lines are near peak - pinch effect
            if ((Bands[i][0][3] > peakPinchPoint[0]) and (Bands[i][0][3] < peakPinchPoint[1])):
                Threshold = pinchThreshold
            else:
                Threshold = cleaveThreshold
            if (Bands[i][0][1] > Threshold):
                cleaveList.append((Bands[i][0][3],Bands[i][0][0],Bands[i][0][1]))
                halfDepth = (int)(Bands[i][0][1]/2)
                halfDensity = (int)(Bands[i][0][2]/2)
                Bands[i][0][1] = halfDepth
                Bands[i][0][2] = halfDensity
                # Synthesize the cleave point
                Bands[i][1][0] = Bands[i][0][0] - halfDepth
                Bands[i][1][1] = halfDepth
                Bands[i][1][2] = halfDensity
                Bands[i][1][3] = Bands[i][0][3]
                Bands[i][1][4] = Bands[i][0][4]
                cleavedCount += 1
        for j in range(Bands_shape[1]):
            if (Bands[i][j][0] == 0):
                break
            depthA += 1
            count += 1
        if (depthA >= Bands_shape[1]):
            print("   depthA exceeds ", Bands_shape[1])
            continue
        bandHist[depthA] += 1
    print("  Cleaved points", cleavedCount)
    print("  count:", count, "depth",depth,"threshold",threshold)

    return cleaveList

def makeGraphFromallLines(allLines, bandTrace, minLength, threshold=0, maxColour=12):
    """ Convert allLines into a np array Trace of lines.

    Parameters:
        allLines: list of lists of lines 
        bandTrace: np array to record lines on
        minLength: minimum length to record
        threshold: check on trace backscatter threshold
        maxColour: maximum value for a point

    Function:
        Convert a list of list of lines into an np array for graphing

    Return Values:
        none
    """

    print("makeGraphFromallLines: minLength", minLength,"threshold",threshold,"maxColour",maxColour)
    #
    # return the number of lines that are greater than minLength
    #
    for i in reversed(range(len(allLines))):
        if (len(allLines[i]) == 0):
            del allLines[i]

    FindLowestPoint = 0
    colour = 1
    maxColour -= 1 # remove the background colour from contention
    longLineNumber = 0
    for i in range(len(allLines)):
        if (len(allLines[i]) < minLength):
            if (allLines[i][0][0] < threshold):
                continue
            else:
                if (len(allLines[i]) < 30):
                    continue
        for j in range(len(allLines[i])):
            if (j+1 >= len(allLines[i])):
                continue
                for k in range(allLines[i][j][1]):
                    bandTrace[allLines[i][j][3]][allLines[i][j][0]-k] = colour
            else:
                if (allLines[i][j][0] > FindLowestPoint):
                    FindLowestPoint = allLines[i][j][0]
                for k in range(allLines[i][j][1]):
                    bandTrace[allLines[i][j][3]][allLines[i][j][0]-k] = colour
        colour += 1
        longLineNumber += 1
        if (colour > maxColour):
            colour = 1
    print("  FindLowestPoint", FindLowestPoint,"last colour",colour)
    return longLineNumber

def makeGraphFromSmoothLines(smoothLines, smoothTrace, noLines, maxColour):
    """ Convert smoothLines into a smoothTrace. 

    Parameters:
        smoothLines: the list of list of smooth (averaged) lines
        smoothTrace: the np array which fills recorded smooth lines
        noLines: number of smooth lines
        maxColour: maximum value for colour

    Function:
        Convert a smoothLines list of lists of line averages into a smoothTrace np array to return.

    Return Values:
        none
    """
    print("makeGraphFromSmoothLines: noLines ",noLines, "maxColour", maxColour)
    # smoothLines 3011 21 7
    FindLowestPoint = 0
    maxColour -= 1

    for line in range(noLines):
        for x in range(len(smoothLines)):
            for z in range(smoothLines[x][line][1]):
                smoothTrace[smoothLines[x][line][3]][smoothLines[x][line][0]-z] = (line%maxColour)+1
    return

def makeGraphFromSmoothLinesAvg(smoothLines, smoothAvgTrace, noLines, maxColour):
    """ Convert smoothLines into a smoothTrace. 

    Parameters:
        smoothLines: the list of list of smooth (averaged) lines
        smoothAvgTrace: the np array which fills recorded smooth lines
        noLines: number of smooth lines
        maxColour: maximum value for colour

    Function:
        Convert a smoothLines list of lists of line averages into a smoothAvgTrace np array around the midpoints to return.

    Return Values:
        none
    """
    print("makeGraphFromSmoothLinesAvg: numLines", noLines, "max colour", maxColour)
    #
    # This requires offest for midPoint
    #
    maxColour -= 1
    for line in range(noLines):
        for x in range(len(smoothLines)):
            offset = (int)(smoothLines[x][line][5]/2)
            for z in range(smoothLines[x][line][5]):
                smoothAvgTrace[smoothLines[x][line][3]][smoothLines[x][line][6]-z+offset] = (line%maxColour)+1
    return

def checkForSpikes(smoothLines, x, line, cleaved):
    """ Attempt to determine the difference between spike artifacts and inclines or declines of the slope of the line.

    Parameters:
        smoothLines: list of list of lines
        x: starting point of the suspect ‘spike’
        Line: line of the ‘spike’
        cleaved: list of cleaved points

    Function:
        Attempt to determine the difference between spike artifacts and inclines or declines of the slope of the line.


    Return Values:
        none
    """
    #print("checkForSpikes", x, line, "y=", smoothLines[x][line][0])
    #
    # Spikes are lines that start either well above or well below the band 
    #   - some of the spikes could be the cleaved points
    #   - if there are too many spikes then it is a step incline/decline and do not adjust
    #
    # Called by PaintBand
    #
    # ToDo:
    #   - perhaps these spikes should be returned to a more appropriate line
    #
    spikeHeight = 20
    spikeNumMax = 15 # How many spikes in a row before increase in slope
    numMovingAvgPoints = 8
    if (x + numMovingAvgPoints > len(smoothLines)):
        return
    numCleaved = 0 # number of spikes that seem to be cleaved
    numJumped = 0  # number of spikes that seem to be line jumps 
    spike = x
    start = x
    baseline = smoothLines[x-1][line][6]+(int)(smoothLines[x-1][line][5]/2)
    spikeDelta = 0

    while (abs(smoothLines[spike][line][0] - baseline) > spikeHeight):
        if ((x + 2 >= len(smoothLines)) or (spike+1 >= len(smoothLines))):
            return
        delta=abs(baseline-smoothLines[spike][line][0])
        spikeDelta += delta
        #
        # Check against cleaved points: (x, y, width)
        #
        cleavedPoint = -1
        for i in range(len(cleaved)):
            if (cleaved[i][0] == spike):
                cleavedPoint = i
                break
        if (cleavedPoint != -1):
            # the spike was due to cleaving
            numCleaved += 1
        #
        # Check if it should be part of another line
        #
        if (line == 0):
            # Check line 1
            otherLine = 1
        #
        # Check whether it's a plateau
        #
        if ((spike-x) > 5):
            spikeAvgWidth = spikeDelta/(spike-x) 
        spike += 1
        if ((spike-x) > spikeNumMax):
            spike = x # reset counter and exit
            break
    if (spike == x): # no spike detected
        return
    #
    # If spikeNumMax is exceeded
    #
    if (spike-start > spikeNumMax):
        return
    spikeDir = 1 # spike Direction increase in y direction, e.g. from 100 to 150 is +ve
    if (smoothLines[start][line][0]-smoothLines[x-1][line][0] < 0):
        spikeDir = -1 # spike Direction is -ve
    for idx in range(start,spike):
        offset=smoothLines[idx][line][0]-smoothLines[x-1][line][0]
     
        avgSpike = (int)(offset/numMovingAvgPoints)
        for z in range(1,numMovingAvgPoints):
            if ((idx+z) >= len(smoothLines)):
                break
            smoothLines[idx+z][line][6] -= (spikeDir * avgSpike)
        smoothLines[idx][line][0] = (int)((smoothLines[x-1][line][0]+smoothLines[spike+1][line][0])/2)
        deSpiked = True
    return

def paintBand(Trace, allLines, minLineLength, numBands=1000):
    """ Convert a list of lists and it’s list of lines to make a Trace. 

    Parameters:
        Trace: the list of list of strengths/backscatter
        allLines: the list of list of lines
        minLineLength: the minimum length of line to consider
        numBands: the maximum number of bands

    Function:
        Convert a list of lists and it’s list of lines to make a graphable Trace.

    Return Values:
        smoothLines - an np array graphically representing bands
    """
    print("paintBand: minimum line length", minLineLength, "numBands", numBands)
    #
    # Convert from list of lists to np.arry in order to graph
    #  Strip lines that are too short
    #  Populate moving averages
    # 
    #  Repair spikes
    #  Attenuate anomolous depths
    #
    # ToDo:
    #  perhaps sort by length: longest to shortest?
    #

    smoothLines = np.zeros((len(Trace), numBands, nData),dtype=int)
    smoothShape = smoothLines.shape

    for i in reversed(range(len(allLines))):
        if (len(allLines[i]) == 0):
            del allLines[i]
    #
    # First - add the allLines list of lists data to the smoothLines np array
    #    
    numMovingAvgPoints = 8
    lineNo = 0 # The line order after short lines have been stripped
    for line in range(len(allLines)):
        lineLen = len(allLines[line])
        if (lineLen == 0):
            continue
        if (lineLen < minLineLength):
            continue
        #
        #
        #
        totalThickness = 0
        avgDepth = [allLines[line][0][1]] * numMovingAvgPoints
        avgMidPoint = [allLines[line][0][0]-(int)(allLines[line][0][1]/2)] * numMovingAvgPoints
        for x in range(len(allLines[line])-1):
            # and if Python had enums: 0-AL_y,1=AL_DEPTH,etc
            # for moving averages, remove first entry and add most recent then re-calculate
            totalThickness += allLines[line][x][1] 
            avgDepth.pop(0)                       # remove first
            avgDepth.append(allLines[line][x][1]) # add most recent
            movingAvg=(int)(sum(avgDepth)/len(avgDepth)) # calculate average Depth
            smoothLines[x][lineNo][5] = movingAvg # calculate average Depth
            smoothLines[x][lineNo][6] = allLines[line][x][0]-(int)(allLines[line][x][1]/2)
            avgMidPoint.pop(0)                                                     # remove first
            avgMidPoint.append(allLines[line][x][0]-(int)(allLines[line][x][1]/2)) # add most recent
            movingAvg=(int)(sum(avgMidPoint)/len(avgMidPoint))                     # calculate average midPoint
            smoothLines[x][lineNo][6] = movingAvg
            smoothLines[x][lineNo][0] = allLines[line][x][0] # y co-ordinate of lowest point
            smoothLines[x][lineNo][1] = allLines[line][x][1] # depth or length of line
            smoothLines[x][lineNo][2] = allLines[line][x][2] # backscatter density of plank (area around point)
            smoothLines[x][lineNo][3] = allLines[line][x][3] # x co-ordinate
            smoothLines[x][lineNo][4] = allLines[line][x][4] # already used
        avgThickness = (float)(((int)(totalThickness*10/len(allLines[line])))/10)
        lineNo += 1
    #
    # Second - attenuate or extend thichkness/depths
    #        - amend stray midpoints
    #
    for line in range(lineNo+1):

        for x in range(2,len(smoothLines)-numMovingAvgPoints): # skip the first, and the last five
            if (smoothLines[x][line][0] == 0):
                break
            altered = False
            checkForSpikes(smoothLines, x, line, cleaveList)

            # Check the depth
            depthDifference = smoothLines[x][line][1] - smoothLines[x][line][5]
            diff = (int)(depthDifference/2)
            if (abs(diff) > 2):
                altered = True

            if (depthDifference < -2):
                smoothLines[x][line][1] += (-1)*depthDifference
            elif (depthDifference > 2):
                smoothLines[x][line][1] -= depthDifference
                
    return smoothLines

def paintVoid(Trace, smoothLines, allLines, numLongLines, minLineLength):
    """ Convert a list of lists and it’s list of lines to find the voids between bands to make a Trace. 

    Parameters:
        Trace: list of lists
        smoothLines: 
        allLines: 
        numLongLines: the number of long lines to consider
        minLineLength: the minimum line lengths to consider 

    Function:
        Convert a list of lists and it’s list of lines to find the voids between bands to make a Trace.

    Return Values:
        smoothLines - list of list
        allVoids - the list of list of Voids
        Vln - the number of long lines
    """
    print("paintVoid: numLongLines", numLongLines, "minimum line length", minLineLength)
    #
    # Paint in the voids in smoothLines
    # Create a list of Voids
    # Voids can only run from end to end (sort of)
    #
    # return smoothLines, allVoids, numLongLines
    #
    smoothShape = smoothLines.shape
    print(" allLine line lengths of",len(allLines),"lines")
    for i in range(len(allLines)):
        if (len(allLines[i]) == 0):
            print("  Forfeiture absconded with line: ", i)
            continue

    for i in range(smoothShape[1]):
        smEnd = 0
        while (smoothLines[smEnd][i][0] != 0):
            smEnd += 1
            if smEnd == len(smoothLines[0]):
                break
        if (smEnd == 0):
            continue
        smEnd -= 1
    #
    # First - add the allLines list of lists data to the smoothLines np array
    #
    numMovingAvgPoints = 8
    allVoids = []
    voidNum = 0
    vln = numLongLines
    for line in range (numLongLines-1):
        initDepth = smoothLines[0][line][0]+1 - smoothLines[0][line+1][0]-smoothLines[0][line+1][1]-1
        avgDepth = [initDepth] * numMovingAvgPoints
        avgMidPoint = [smoothLines[0][line][0]+(int)(initDepth/2)] * numMovingAvgPoints
        vln = numLongLines+voidNum # void+line number
        voidLine = []
        for x in range(len(smoothLines)):
            skipLine = False
            loff = 1  #  line offset
            priorIdx1 = idx1 = idx2 = x
            while ((smoothLines[idx1][line][3] != smoothLines[idx2][line+loff][3]) and (line+loff < numLongLines)):
                if (smoothLines[idx1][line][3] >= smoothLines[0][line+loff][3]):
                    idx2 = 0
                    while ((smoothLines[idx1][line][3] != smoothLines[idx2][line+loff][3]) and (idx2 < len(smoothLines)-1)):
                         idx2 += 1
                    if (idx2 == len(smoothLines)-1):
                        loff += 1
                        continue
                    break
                else:
                    if (smoothLines[idx1][line][3] == smoothLines[idx2][line+loff+1][3]):
                        loff += 1
                    else:
                        if ((priorIdx1+1 < len(smoothLines)-1)and(smoothLines[priorIdx1+1][line][3] == smoothLines[idx2][line+loff][3])):
                            idx1 = priorIdx1+1
                        else:
                            while ((smoothLines[idx1][line][3] != smoothLines[idx2][line+loff][3]) and (idx1 < len(smoothLines)-1)):
                                idx1 += 1
                                if ((smoothLines[idx1][line][3] == 0) and (smoothLines[idx1][line][0] == 0)):
                                    skipLine = True
                                    break
                    break
                        
            if (line+loff >= numLongLines):
                break
            if (skipLine == True):
                break
            yHigh = smoothLines[idx1][line][0]+1
            yLow = smoothLines[idx2][line+loff][0]-smoothLines[idx2][line+loff][1]-1
            depth = yLow - yHigh
            density = 0
            for y in range(yHigh,yLow+1):
                density += abs(Trace[x][y])
            midPoint = yHigh+(int)(depth/2)
            if (yLow > 200) and (yHigh > 200):  # Must figure out
                avgDepth.pop(0)        # remove first
                avgDepth.append(depth) # add most recent
                newDepthAvg = (int)(sum(avgDepth)/len(avgDepth))
                avgMidPoint.pop(0)           # remove first
                avgMidPoint.append(midPoint) # add most recent
                newMidPointAvg = (int)(sum(avgMidPoint)/len(avgMidPoint))
            else:
                newDepthAvg = (int)(sum(avgDepth)/len(avgDepth))
                newMidPointAvg = (int)(sum(avgMidPoint)/len(avgMidPoint))
            if (yLow < 200) or (yHigh < 200):
                continue
            if (depth <= 0):
                depth = 1
                yHigh = yLow-1
                midPoint = yHigh
            if (yLow < 200):
                continue
                

            smoothLines[x][vln][0] = yLow # y co-ordinate of lowest point
            smoothLines[x][vln][1] = depth # depth or length of line
            smoothLines[x][vln][2] = density # backscatter density of line
            smoothLines[x][vln][3] = smoothLines[idx1][line][3] # x co-ordinate 
            smoothLines[x][vln][4] = 0 # already used?
            smoothLines[x][vln][5] = (int)(sum(avgDepth)/len(avgDepth))
            smoothLines[x][vln][6] = (int)(sum(avgMidPoint)/len(avgMidPoint))
            voidLine.append([yLow,depth,density,x,0,newDepthAvg,newMidPointAvg])
            priorIdx1 = idx1
        allVoids.append(voidLine)
        voidNum += 1
            
    return smoothLines,allVoids,vln

def Stitcher(Trace, allLines, minLineLength, maxNumLines, XStitchThreshold=25, YStitchThreshold=60):
    """ Bind line fragments into longer lines. 

    Parameters:
        Trace: list of list of strengths/backscatter
        allLines: the list of lines
        minLineLength: the minimum length of a line fragment
        maxNumLines: the maximum number of lines
        XStitchThreshold: the X radius to match lines 
        YStitchThreshold: the Y radius to match lines

    Function:
        Bind the end of a line fragment to the start of another line if it is within the radius of x,y thresholds

    Return Values:
        none
    """
    print("Stitcher: allLines len:", len(allLines),"minLineLength",minLineLength,"maxNumLines",maxNumLines)
    print("         thresholds: X",XStitchThreshold," Y",YStitchThreshold)
    # 
    # Stitcher combines lines that terminate and start nearby each other
    #     The mechanism is clunky as it restarts after every line combination
    #
    # The code returns a modified allLines
    #
    list_len = [len(i) for i in allLines]
    while True:
        altered = False
        for i in range(len(allLines)-1):
            lineLen = allLines[i][len(allLines[i])-1][3] - allLines[i][0][3]
            if (lineLen+XStitchThreshold > len(Trace)):
                continue
            for j in range(i+1,len(allLines)-1):
                lineLen = allLines[j][len(allLines[j])-1][3] - allLines[j][0][3]
                if (lineLen+XStitchThreshold > len(Trace)):
                    continue
                deltaX = allLines[j][0][3] - allLines[i][len(allLines[i])-1][3] 
                deltaY = allLines[i][len(allLines[i])-1][0] - allLines[j][0][0]
                Accepted = False
                if (deltaX >= 0) and ((deltaX > XStitchThreshold) or (abs(deltaY) > YStitchThreshold)):
                    # This would have been rejected under old schema
                    if ((deltaX <= YStitchThreshold) and (abs(deltaY) <= XStitchThreshold)):
                        Accepted = True
                else:
                    if ((deltaX < 0) or (deltaX > XStitchThreshold)):
                        Accepted = False
                    else:
                        if (abs(deltaY) < YStitchThreshold):
                            Accepted = True
                if (Accepted == True):
                    #
                    # Synthesize the points
                    #
                    startX = allLines[i][len(allLines[i])-1][3]+1
                    endX = allLines[j][0][3]
                    sizeX = endX-startX
                    if (sizeX > 0): 
                        incY = (int)(deltaY/sizeX)
                        curWidth = allLines[i][len(allLines[i])-1][1]
                        widthDelta = (int)((allLines[j][0][1] - curWidth)/sizeX)
                        curY = allLines[i][len(allLines[i])-1][0]
                        for k in range(startX,endX):
                            curY -= incY
                            curWidth += widthDelta
                            synthLine=(curY,curWidth,allLines[i][0][2],k,False,allLines[i][0][4],allLines[i][0][5])
                            allLines[i].append(synthLine)
                    allLines[i].extend(allLines[j])
                    del allLines[j]
                    altered = True
                    break
            if (altered == True):
                break
        if (altered == False):
            break

    print(" After Stitcher: allLines len:", len(allLines))
    list_len = [len(i) for i in allLines]
    print("   list_len", list_len)
   
    return allLines

def Wrangler(allLines):
    """ Tries to restore the order of the lines. 

    Parameters:
        allLines: a list of list of lines

    Function:
        Wrangler applies the Time Team axiom that the top layers are younger than the lower layers, excepting folding, which is quite possible, bands will be deposited over time so when the lines cross it is most likely a touch but this confuses the algorithm, Wrangler restores the lines being deposited in order.

    Return Values:
        allLines, a list of list of lines
    """
    print("Wrangler:")
    # 
    # Wrangler applies the Time Team axiom that the top layers are younger than the lower layers
    #   Excepting folding, which is quite possible, bands will be deposited over time so when the
    #   lines cross it is most likely a touch but this confuses the algortihm - Wrangler restores
    #   the lines a being deposited in order.
    #
    # The allLines structure is a list of lists of lists - a list of lines of (y,depth,backscatter,x,used,avgDepth,avgMidPoint)
    #
    # Note: moving averages have not been recalculated
    #
    # The code returns a modified allLines
    #
    numLines = 0
    for i in range(len(allLines)):
        if (len(allLines) == 0):
            break
        numLines += 1
    print(" numLines", numLines)
    if (numLines == 1):
        print("  Only a single line found - Wrangling not required")
        return allLines
    
    for target in range(numLines-1):
        for line in range(target+1,numLines):
            idx1 = 0
            idx2 = 0
            while (idx1 < len(allLines[target])) and (idx2 < len(allLines[line])):
                x1 = allLines[target][idx1][3]
                x2 = allLines[line][idx2][3]
                # determine which line segement to advance
                if (x1 < x2):
                    while(allLines[target][idx1][3] < x2):
                        idx1 += 1
                        if (idx1 == len(allLines[target])):
                            idx1 = -1
                            break
                        x1 = allLines[target][idx1][3]
                else:
                    while (allLines[line][idx2][3] < x1):
                        idx2 +=1
                        if (len(allLines[line]) == idx2):
                            idx2 = -1
                            break
                        x2 = allLines[line][idx2][3]
                if (idx1 == -1) or (idx2 == -1):
                    break
                y1 = allLines[target][idx1][0]
                y2 = allLines[line][idx2][0]
                if (y1 > allLines[line][idx2][0]):
                    aline = allLines[target][idx1]
                    bline = allLines[line][idx2]

                    allLines[target][idx1] = bline
                    allLines[line][idx2] = aline

                idx1 += 1
                idx2 += 1

    return allLines

def Forfeiture(Trace, allLines, lawLine=0):
    """ Move parts of shorter lines to parallel and close longer lines. 

    Parameters:
        Trace: list of list of strengths/backscatter 
        allLines: a list of list of lines
        lawLine: minimum number of lines 

    Function:
        Usurp parts of lines to make longer single lines. This became necessary as Stitcher could not join lines that are parallel but would overlap.

    Return Values:
        allLines (updated)
    """
    print("Forfeiture: lawLine", lawLine)
    # 
    # Forfeiture confiscating other lines points 
    #
    # The allLines structure is a list of lists of lists - a list of lines of (y,depth,backscatter,x,used,avgDepth,avgMidPoint)
    #
    # Note: moving averages have not been recalculated
    #
    # The code returns a modified allLines
    #
    if (len(allLines) == 1):
        print("  Only a single line found - Forfeiture not required")
        return allLines
    if (len(allLines) < lawLine):
        return allLines
    
    numLines = 0
    limit = (int)(len(Trace)*0.4)
    margin = (int)(len(Trace)*0.02)
    for i in range(len(allLines)):
        if (len(allLines[i]) == 0):
            continue
        numLines += 1
        if (len(allLines[i])-1 < limit):
            continue
        lawLine = i

        startX = allLines[lawLine][0][3]
        stopX = allLines[lawLine][len(allLines[lawLine])-1][3]
        stopIdx = len(allLines[lawLine])-1

        alteredStart = False
        alteredEnd = False
        for target in range(numLines-1):
            if (target == lawLine):
                continue
            if (len(allLines[target]) == 0): # target has been stripped of length by previous forfeiture
                continue
            if (len(allLines[target]) + margin > len(Trace)): #if the target line is near complete - don't touch
                continue
            if (stopX < allLines[target][0][3]):
                continue
            if (len(allLines[target]) == 0): # target has been stripped of length by previous forfeiture
                del allLines[target]
                continue
            #
            # Check if the start ofthe target line can be taken.
            # Note that the lines must be parallel at some point otherwise
            #     a valid combining would have occurred in Stitcher()
            #
            #
            
            #if (allLines[target][len(allLines[target])-1][3] < startX):
            #    print("should be rejected as falling short")
            idx = 0
            if ((allLines[target][idx][3] < startX) and (allLines[target][len(allLines[target])-1][3] < startX) and (alteredStart == False)):
                alteredStart = True
                targetLen =  len(allLines[target])-1
                diff = startX-allLines[target][idx][3]
                allLines[lawLine][0:1] = allLines[target][idx:idx+diff]
                del allLines[target][:diff]
                startX = allLines[lawLine][0][3]
                stopX = allLines[lawLine][len(allLines[lawLine])-1][3]
                
            if (len(allLines[target]) < 1):
                continue
            idx = len(allLines[target])-1
            if ((allLines[target][idx][3] > stopX) and (alteredEnd == False)):
                alteredEnd = True
                diff = allLines[target][idx][3] - stopX
                allLines[lawLine][stopX:stopX+1] = allLines[target][idx:idx+diff]
                del allLines[target][0:diff]

    print(" return allLines of len", len(allLines))
    return allLines

def Mortar(allLines, minLineLength, XMortThreshold=40, YMortThreshold=150,bedrock=True):
    """ Combine short, thick bedRock type lines. 

    Parameters:
        allLines: list of lists of lines 
        minLineLength: minimum line length
        XMortThreshold: x distance that can be bridged 
        YMortThreshold: y distance that can be bridged
        Bedrock: flag indicating the bottom is len(Trace[0]) 

    Function:
        For short, sloping lines, but the lines together while being able to join overlapping lines (unlike Stitcher).

    Return Values:
        allLines (updated)
    """
    print("Mortar: lines", len(allLines),"minLength",minLineLength," thresholds X,Y",XMortThreshold,YMortThreshold)
    # 
    # Mortar combines short lines by adding a layer between the lines
    #     This is a first stab a developing feature identification
    #
    # The code returns a modified allLines
    #
    list_len = [len(i) for i in allLines]
    idx = 0 # points to the bottom line
    startNumLines = len(allLines)
    bedrockThreshold = int(len(Trace[0])*0.98)
    startBedrock = False
    startBedrockAtX = -1
    inBedrock = False
    yDirection = -1 # Y axis upwards

    while True:
        if (idx+1 == len(allLines)):
            break
        if (bedrock==True) and (startBedrock==False):
            bedrockIdx = -1
            for i in range(len(allLines[idx])):
                if (allLines[idx][i][0] > bedrockThreshold):
                    bedrockIdx = i
                    startBedrockAtX = allLines[idx][i][3]
                    break
            if (bedrockIdx == -1):
                del allLines[idx]
                continue
            startBedrock = True
            if (bedrockIdx != 0):
                for i in range(bedrockIdx):
                    allLines[idx].pop(0)
        #
        # Compare the dimensions of the two lines
        #
        # Bottom line dimensions
        leftXB = allLines[idx][0][3]
        riseA = allLines[idx][0][0]
        riseB = allLines[idx][len(allLines[idx])-1][0]
        slope=(riseA-riseB)/len(allLines[idx])
        endLineLen  = 9
        if len(allLines[idx]) < 9:
            endLineLen = len(allLines[idx])
        riseA = allLines[idx][len(allLines[idx])-endLineLen-1][0]
        riseB = allLines[idx][len(allLines[idx])-1][0]
        endSlope=(riseA-riseB)/endLineLen
        rightXB = allLines[idx][len(allLines[idx])-1][3]
        lowYB = allLines[idx][0][0]
        if (lowYB > bedrockThreshold):
            if (startBedrock == False):
                startBedrockAtX = allLines[idx][0][0]
                startBedrock = True
        highYB = allLines[idx][0][0]-allLines[idx][0][1]
        for i in range(1,len(allLines[idx])):
            if (allLines[idx][i][0] > bedrockThreshold):
                if (startBedrock == False):
                    startBedrockAtX = leftXB
                    startBedrock = True
            if (lowYB > allLines[idx][i][0]):
                lowYB = allLines[idx][i][0]
            if (highYB > (allLines[idx][i][0] - allLines[idx][i][1])):
                highYB = allLines[idx][i][0] - allLines[idx][i][1]
        # Top line dimensions
        leftXT = allLines[idx+1][0][3]
        rightXT = allLines[idx+1][len(allLines[idx+1])-1][3]
        lowYT = allLines[idx+1][0][0]
        highYT = allLines[idx+1][0][0]-allLines[idx+1][0][1]
        for i in range(1,len(allLines[idx+1])):
            if (lowYT < allLines[idx+1][i][0]):
                lowYT = allLines[idx+1][i][0]
            if (highYT > (allLines[idx+1][i][0] - allLines[idx+1][i][1])):
                highYTB = allLines[idx+1][i][0] - allLines[idx+1][i][1]
        # Compare the dimensions
        if ((leftXB > rightXT) and ((leftXB - rightXT) > XMortThreshold)):
            idx += 1
            continue
        if ((rightXB < leftXT) and ((rightXB - leftXT) > XMortThreshold)):
            idx += 1
            continue
        deltaY = lowYT - highYB
        if (deltaY > YMortThreshold) and (yDirection > 0):
            idx += 1
            continue
        #
        #  Threshold vetting for prepend and append - look at line ends and beginnings
        #
        if (leftXB > rightXT):  # too far left - prepend
            delta3 = allLines[idx][0][0]-allLines[idx+1][len(allLines[idx+1])-1][0]
            delta2 = allLines[idx][0][0]-allLines[idx+1][len(allLines[idx+1])-1][0]
            delta1 = allLines[idx][0][0]-allLines[idx+1][len(allLines[idx+1])-1][0]
            if (abs(delta1) > YMortThreshold) or (abs(delta2) > YMortThreshold) or (abs(delta3) > YMortThreshold):
                idx += 1
                if (startBedrock  == True):
                    startBedrock = False
                continue
        elif (rightXB < leftXT): # too far right - append
            delta3 = allLines[idx][len(allLines[idx])-3][0]-allLines[idx+1][0][0]
            delta2 = allLines[idx][len(allLines[idx])-2][0]-allLines[idx+1][0][0]
            delta1 = allLines[idx][len(allLines[idx])-1][0]-allLines[idx+1][0][0]
            if (abs(delta1) > YMortThreshold) or (abs(delta2) > YMortThreshold) or (abs(delta3) > YMortThreshold):
                idx += 1
                if (startBedrock  == True):
                    startBedrock = False
                continue
        else: # intersect - could require prepend and/or append, and could be inverted
            if (rightXB < rightXT): # Check for append
                targetIdx = 0 # the last point that lines match
                for i in range(len(allLines[idx+1])):
                    if (allLines[idx+1][i][3] == allLines[idx][len(allLines[idx])-1][3]):
                        targetIdx = i
                        break
                if (len(allLines[idx])-3 < 0):
                    break
                delta3 = allLines[idx+1][targetIdx][0]-allLines[idx][len(allLines[idx])-3][0]
                delta2 = allLines[idx+1][targetIdx][0]-allLines[idx][len(allLines[idx])-2][0]
                delta1 = allLines[idx+1][targetIdx][0]-allLines[idx][len(allLines[idx])-1][0]
                if (abs(delta1) > YMortThreshold) or (abs(delta2) > YMortThreshold) or (abs(delta3) > YMortThreshold):
                    idx += 1
                    if (startBedrock  == True):
                        startBedrock = False
                    continue
            if (leftXB > leftXT):   # Check for prepend
                if (len(allLines[idx])-3 < 0):
                    delta1 = allLines[idx][len(allLines[idx])-1][0]-allLines[idx+1][0][0]
                    if (abs(delta1) > YMortThreshold):
                        idx += 1
                        continue
                else:
                    delta3 = allLines[idx][len(allLines[idx])-3][0]-allLines[idx+1][0][0]
                    delta2 = allLines[idx][len(allLines[idx])-2][0]-allLines[idx+1][0][0]
                    delta1 = allLines[idx][len(allLines[idx])-1][0]-allLines[idx+1][0][0]
                    if (abs(delta1) > YMortThreshold) or (abs(delta2) > YMortThreshold) or (abs(delta3) > YMortThreshold):
                        idx += 1
                        if (startBedrock  == True):
                            startBedrock = False
                        continue
        #
        #
        # Determine if this is Bedrock
        if (lowYT > bedrockThreshold):
            if (startBedrock == False):
                startBedrockAtX = leftXB
            startBedrock = True
        if (startBedrock == False):
            idx += 1
            continue
        yFloor = yHeight = -1
        deltaX = int(round(((rightXB-leftXB) + (rightXT-leftXT))/2))

        #
        # Determine how to process idx+1
        #
        if (leftXB > rightXT):  # too far left - prepend
            # determine gap filler
            lastY= allLines[idx+1][len(allLines[idx+1])-1][0]
            heightLastY= allLines[idx+1][len(allLines[idx+1])-1][1]
            # prepend the bottom line
            for i in range(len(allLines[idx+1])-1):
                allLines[idx].insert(i,allLines[idx+1][i])
            # fill in gaps
            fillIdx = len(allLines[idx+1])
            for i in range(rightXT+1,leftXB):
                line=[lastY,heightLastY,0,i,0,0,0]
                allLines[idx].insert(fillIdx, line)
            del(allLines[idx+1])
        elif (rightXB < leftXT): # too far right - append
            #
            # determine gap filler
            lastY= allLines[idx][len(allLines[idx])-1][0]
            heightLastY= allLines[idx][len(allLines[idx])-1][1]
            # append gap filler to the bottom line
            fillIdx = len(allLines[idx])
            for i in range(rightXB+1,leftXT):
                line=[lastY,heightLastY,0,i,0,0,0]
                allLines[idx].append(line)
            for i in range(len(allLines[idx+1])):
                allLines[idx].append(allLines[idx+1][i])
            # fill in gaps
            del(allLines[idx+1])
        else: # intersect - could require prepend and/or append, and could be inverted - oh joy
            # Check for append
            if (rightXB < rightXT):
                fillIdx = 0
                for i in range(len(allLines[idx+1])-1):
                    if (allLines[idx+1][i][3] == rightXB):
                        fillIdx = i+1
                        break
                if (fillIdx == 0):
                    print("  intersection append fillIdx not found")
                else:
                    for i in range(fillIdx,len(allLines[idx+1])):
                        allLines[idx].append(allLines[idx+1][i])

            # Check for prepend
            if (leftXB > leftXT):
                fillIdx = 0       
                for i in range(leftXT,leftXB):
                    allLines[idx].insert(fillIdx,allLines[idx+1].pop(0))
                    fillIdx += 1

            # Amend for intersection - beware inversion
            for i in range(len(allLines[idx])-1):
                for j in range(len(allLines[idx+1])-1):
                    if (allLines[idx][i][3] == allLines[idx+1][j][3]):
                        # check for inversion
                        deltaY = allLines[idx][i][0] - allLines[idx+1][j][0]
                        if (abs(deltaY) > YMortThreshold):
                            continue
                        newDelta = abs(deltaY) - allLines[idx][i][1] + allLines[idx+1][j][1]
                        if (newDelta < 0):
                            continue
                        if (deltaY < 0): # inverted
                            allLines[idx][i][0] = allLines[idx+1][j][0]
                            allLines[idx][i][1] = abs(deltaY) + allLines[idx+1][j][1]
                        else:
                            allLines[idx][i][1] = deltaY + allLines[idx+1][j][1]
                            
            del(allLines[idx+1])

    print(" After Mortar: allLines len:", len(allLines))
    list_len = [len(i) for i in allLines]
    print(" list_len", list_len)
    bottom=len(Trace[0])-1
    for i in range(len(allLines)):
        if (bedrock == True):
            for k in range(len(allLines[i])):
                delta = bottom-allLines[i][k][0]+allLines[i][k][1]
                allLines[i][k][0] = bottom
                allLines[i][k][1] = delta

        
    return allLines

def Analyse(Trace, ystart=800, ystop=200, threshold=1200, nBands=61, thin=False):
    """ Search for bands and lines. 

    Parameters:
        Trace: list of lists of backscatter/strengths
        Ystart: y starting point going upwards
        Ystop: y stopping point
        Threshold: the backscatter/strength indicating a band piece
        nBands: maximum number of bands
        thin: flag indicating thin bedrock lines

    Function:
        Scan a group of horizontal lines from Ystart to Ystop in Trace looking for rectangles (5x7 points) containing ‘threshold’ value of backscatter / strength. This subroutine calls lookForBands(), cleave(), and lookForLines() and, depending on the bedrock flag would call either Mortar() and perhaps parabola() or Wrangler(), Stitcher(), and Forfeiture(). 

    Return Values:
        (allLines, cleaveList)
    """
    print("Analyse: ystart", ystart," ystop", ystop, "threshold", threshold,"nBands", nBands)
    #
    # Analyse() is the first step towards modularization of band detection
    #   Creates Bands, 
    #   calls LookForBand(), cleave(), and LookForLines()
    #
    
    nData=7 #y,depth,energy,x,used,avg5depth,avg5midpoint
    Bands = np.zeros((len(Trace), nBands, nData),dtype=int)
    Bands_shape = Bands.shape
    maxNumLines = (int)(Bands_shape[0]*0.30)   # arbitrary limit
    if (thin == True):
        minLineLength = 5
        depth=4  # the depth of the band
        width=7  # the width of the band
    else:
        minLineLength = (int)(Bands_shape[0]*0.015) # arbitrary limit
        depth=5  # the depth of the band
        width=7  # the width of the band
    print("  maxNumLines",maxNumLines,"minLineLength",minLineLength,"thin",thin)

    #
    # Magic - until findBedrock() returns these values
    #
    peakPinchPoint = [2080,2140] # BIG HONKING HACK as the peak pinches the lines
                                 #   this migth be a parameter filled in by find bedrock
    pinchThreshold = 10
    cleaveThreshold = 25

    #
    # Parameters for this round of band searching
    #
    yp = ystart         # the y point from which to start searching
    #
    # Checking for band at a point checks width/2 points to the right and left to
    #   check for threshold amount of backscatter.
    #
    depth=4  # the depth of the band 
    #depth=5  # the depth of the band
    width=7  # the width of the band
    #
    # Look for band
    #
    lookForBand(Trace, Bands, yp, ystop, depth, width, threshold)
    #
    # Look for single cleave points
    #
    print("")
    cleaveList = [] # list of tuples (x, y, width)
    cleave(Bands, Bands_shape, threshold, cleaveList, cleaveThreshold, peakPinchPoint, pinchThreshold)
    print(" cleave returns cleaveList",len(cleaveList))
    if (len(cleaveList) < 10 and len(cleaveList) > 0):
        print("  cleavelist:",cleaveList)
    #
    # Count the number of points in the Bands np array
    #
    lines = [] * maxNumLines
    countPoints = 0
    usedPoints = 0
    for i in range(Bands_shape[0]):
        for j in range(Bands_shape[1]):
            if (Bands[i][j][0] != 0):
                countPoints += 1
            if (Bands[i][j][4] == True):
                usedPoints += 1
    print(" countPoints",countPoints," of",Bands_shape[0]*Bands_shape[1])
    print("   used point - should be zero", usedPoints)
    print("")
    #
    # lookForLines - given an np array, try to rationalize these points into lines
    #
    lookLineStartTime = datetime.datetime.now()
    allLines = lookForLines(Bands,nBands,maxNumLines,minLineLength)
    lookLineEndTime = datetime.datetime.now()
    print(" timex lookForLines:", lookLineEndTime-lookLineStartTime)
    ##
    print("  Analyse - allLines len:", len(allLines))
    list_len = [len(i) for i in allLines]
    if (len(list_len) < 40):
        print("   list_len", list_len)
    
    return (allLines,cleaveList)
 
def makeStatsFromallLines(allLines, minLength, Trace, phaseTrace, differential=False, Tag=""):
    """ Create tables of statistics. 

    Parameters:
        allLines: list of lists of lines
        minLength: minimum line length 
        Trace: list of lists of backscatter/strengths
        phaseTrace: list of lists of phase lengths
        Differential: flag to accommodate bands estimated using polyfit
        Tag: label 

    Function:
        Print statistics on length, area, and density

    Return Values:
        none
    """
    print("makeStatsFromallLines: ",Tag," minLength", minLength, " differential", differential)
    #
    if(len(allLines) == 0):
        print("  received empty allLines, returning")
        return

    print(" allLine lengths of",len(allLines),"lines")
    for i in reversed(range(len(allLines))):
        if (len(allLines[i]) == 0):
            del allLines[i]

    FindLowestPoint = 0
    for i in range(len(allLines)):
        if (len(allLines[i]) < minLength):
            continue
        highY = 0
        lowY = len(Trace[0])
        linePhaseTotal = 0
        offLinePhaseTotal = 0
        totalTraces = 0    # the contiguous X points 
        totalDepth  = 0 # total strengths - effectively the area in pixels
        totalStrength = 0  # backscatter
        totalInLineStrength = 0 # ten points in the line
        totalOffLineStrength = 0 # ten points above the line
        totalRecLength = 0 # for differential don't record areas estimated using polyfit and parabola()
        y = allLines[i][0][0]-allLines[i][0][1]
        if (y > 1500):# should be parameter but running out of time
            differentialThreshold = 1500 # should sample up from top of line
        else:
            differentialThreshold = 200 # should sample down from the bottom of line
    
        for j in range(len(allLines[i])):
            totalTraces += 1
            if (allLines[i][j][0] > highY):
                highY = allLines[i][j][0]
            if (allLines[i][j][0]-allLines[i][j][1] < lowY):
                lowY = allLines[i][j][0] - allLines[i][j][1]
            totalDepth += allLines[i][j][1]
            yt = allLines[i][j][0]-allLines[i][j][1]
            yb = allLines[i][j][0]
            if (differential == True and yt > differentialThreshold):
                x = allLines[i][j][3]
                totalRecLength += 1
                if (differentialThreshold == 200):
                    linePhase = phaseTrace[x][yb]
                else:
                    linePhase = phaseTrace[x][yt]
                yp = yt
                while (linePhase == 0): # This happens in the lower region
                    yp -= 1 # technically this should be +ve for upper regions
                    linePhase = phaseTrace[x][yp]
                    if (yp == 0):
                        break
                if (differentialThreshold == 200):
                    yStart = yb + 3
                    yEnd = yb + 13
                else:
                    yStart = yt - 13
                    yEnd = yt - 3
                for z in range(yStart, yEnd):
                    totalOffLineStrength += abs(Trace[x][z])
                for z in range(y, y+10):
                    if z > len(Trace[0])-1:
                        break
                    totalInLineStrength += abs(Trace[x][z])

                while linePhase == phaseTrace[x][y] and phaseTrace[x][y] != 0:
                    y -= 1
                    if y <= 0:
                        break
                linePhaseTotal += linePhase
                offLinePhaseTotal += phaseTrace[x][y]
            for k in range(allLines[i][j][1]):
                x = allLines[i][j][3]
                y = allLines[i][j][0]-k
                if (y > len(Trace[0])-1):
                    print("  OOOPSS y",y," reset to",len(Trace[0])-1)
                    y = len(Trace[0])-1
                totalStrength += abs(Trace[x][y])
        length = round(len(allLines[i]) * hd["stepSize"])
        area = round(totalDepth/1000*hd["areaPerSample"],2) # square kilometers
        print(" ",Tag,": ",i,"len",length,hd["posUnits"]," Y h/l",highY,lowY,"area",area,"sq. km. avgDepth",round(totalDepth/len(allLines[i]),2),"Strength",totalStrength,"strDensity",round(totalStrength/totalDepth,2))
        lineLen = len(allLines[i])
        if (differential == True):
            print("    linePhaseTotal",linePhaseTotal,"totalRecLength",totalRecLength)
            print("     Phase avg %.2f" % (linePhaseTotal/totalRecLength)," off %.2f" % (offLinePhaseTotal/totalRecLength))
            print("     Line str %.2f" % (totalInLineStrength/(totalRecLength*10))," off %.2f" % (totalOffLineStrength/(totalRecLength*10)))
            
    return

def cycleFill(k,startIdx,stop,center,maxCycleLen,phases,phaseData,invPhaseData,phaseDecadeData,cycleHist,cycleRegHist,decadeHist,decadeRegHist,Tag):
    """ Applies the period gain table to the data. 

    Parameters:
        k: x value
        startIdx: the start of the period
        stop: the last point of the period
        Center: the center point of the period
        maxCycleLen: the maximum possible length of a period
        Phases: table of arcsines
        phaseData: the np array of arcsine adjusted values
        invPhaseData: the np array of the phase lengths
        phaseDecadeData: np array of more precise phase lengths
        cycleHist: histogram of period lengths
        cycleRegHist: histogram of eight regions period lengths
        decadeHist: histogram of precise period lengths
        decadeRegHist: histogram of eight regions precise period lengths
        Tag: identifier of the calling routine

    Function:
        Ok, it’s a bit of a bodge, as it was developed late and expanded greatly.

    Return Values:
        none
    """

    cycleLen = stop - startIdx + 1
    if cycleLen < 2:
        print("Tag ",Tag," ",k,startIdx,":phase ",Trace[k][startIdx]," look ",Trace[k][startIdx-5:stop+5])
    
    if center - startIdx < 0: # problem with initial phase near time zero
        for z in range(startIdx,stop):
            if (Trace[k][z] < 0):
                if (center != z):
                    center = z
                break
        if center - startIdx < 0: # problem with initial phase near time zero
            center = startIdx + int((stop-startIdx)/2)

    front=abs(Trace[k][startIdx-1]) + abs(Trace[k][startIdx])
    if (front == 0):
        front = 1
    frontL=int(abs(Trace[k][startIdx-1])*10/front)
    frontR=int(abs(Trace[k][startIdx])*10/front)
    back=abs(Trace[k][stop-1]) + abs(Trace[k][stop])
    if (back == 0):
        back = 1
    backL=int(abs(Trace[k][stop-1])*10/back)
    backR=int(abs(Trace[k][stop])*10/back)
    decade = (stop-startIdx)*10+frontR+backL

    for n in range(startIdx, stop):
        if (cycleLen >= 16):
            phaseData[k][n] = 16
            phaseDecadeData[k][n] = decade
        else:
            phaseData[k][n] = int(decade/10)
            phaseDecadeData[k][n] = decade
            
    # Note:
    #  Most periods begin with a positive rising value and end with a negative rising value - as close
    # to a 2 * pi period as possible *but* the real world gpr traces are not ideal and whole periods
    # can exist above or below the zero line. So the abnormally long lines are cut in unWave() and may
    # result in entire periods without positives or negatives, so the code must change to accomodate
    phaseIdx = (center - startIdx)*2
    idx = 0
    total = 0
    for n in range(startIdx, center):
        if (phaseIdx >= len(phases) or (idx > len(phases[phaseIdx]))):
            print("center - startIdx",center - startIdx,"startIdx, center ",startIdx, center," k,n",k,n,"phaseIdx,idx",phaseIdx,idx,"len phases",len(phases))
            continue
        invPhaseData[k][n] = int(phases[phaseIdx][idx] * abs(Trace[k][n]))
        if (invPhaseData[k][n] == 0) and (n == startIdx):
            # the starting point is a zero - try to adjust
            invPhaseData[k][n] = int((invPhaseData[k][n-1] + abs(Trace[k][n+1]))/2)
        total += invPhaseData[k][n]
        idx += 1
        # If the center value is zero make avg of positives
        if (n == (center - 1) and invPhaseData[k][n] == 0):
            if (idx == 1):
                idx += 1
            invPhaseData[k][n] = int(total/(idx-1))
    phaseIdx = (stop - center)*2
    idx = 0
    for n in range(center, stop):
        if (Trace[k][n] < 0):
            if (phaseIdx >= maxCycleLen):
                print(" HERE max exceeded", maxCycleLen,"k,n",k,n," phaseIdx,idx",phaseIdx,idx)
                phaseIdx = maxCycleLen - 1
            invPhaseData[k][n] = int(phases[phaseIdx][idx] * abs(Trace[k][n]))
        elif (Trace[k][n] == 0):
            invPhaseData[k][n] = int(phases[phaseIdx][idx] * int((abs(Trace[k][n-1]) + abs(Trace[k][n+1]))/2))
        else:
            invPhaseData[k][n] = int(phases[phaseIdx][idx] * abs(Trace[k][n]))
        idx += 1
    # Vet for Negatives
    negCount = 0
    for n in range(startIdx,stop):
        if (invPhaseData[k][n] < 0):
            negCount += 1
    if negCount > 0:
        print("  k:",k,"negCount",negCount,"startIdx:stop",startIdx,stop,":",invPhaseData[k][startIdx:stop])
        
    # record cycle history
    zP = int(hd["timeZero"])
    regionLen = (len(Trace[0])-int(hd["timeZero"]))/8
    if cycleLen > 1:
        if (cycleLen > len(cycleHist)-1):
            print(" len cycleHist",len(cycleHist)," cycleHist", cycleHist)
            cycleHist[len(cycleHist)-1] += 1
            print(" len cycleRegHist",len(cycleRegHist),len(cycleRegHist[0])," ", int((startIdx-zP)/regionLen), len(cycleHist)-1)
            cycleRegHist[int((startIdx-zP)/regionLen)][len(cycleHist)-1] += 1
            decadeRegHist[int((startIdx-zP)/regionLen)][len(decadeHist)-1] += 1
        else:
            cycleHist[cycleLen] += 1
            cycleRegHist[int((startIdx-zP)/regionLen)][cycleLen] += 1
            decadeRegHist[int((startIdx-zP)/regionLen)][decade] += 1
            if (Tag == "I"):
                decadeHist[decade] += 1
    return

def unWave(Trace, maxValue):
    """ Gain function which corrects for the wave nature of the signal. 

    Parameters:
        Trace: list of lists of backscatter/strengths
        maxValue: used for graphs called within function (fix)

    Function:
        Build a list of arcsine values based on cycle length, and multiply the strengths by the arcsines to correct them.

    Return Values:
        phaseData,invPhaseData
    """
    print("unWave: maxValue", maxValue)
    phaseDecadeData = np.zeros([len(Trace), len(Trace[0])], int)
    phaseData = np.zeros([len(Trace), len(Trace[0])], int)
    invPhaseData = np.zeros([len(Trace), len(Trace[0])], int)
    #
    # Build a list of arcsine values based on cycle length
    # this is so multiple millions of trig functions are not needed
    # also, only need a half cycle
    #
    maxCycleLen = 53 # always choose primes as magic numbers - imo
    phases = [[1],[1]]
    for m in range(2,maxCycleLen):
        thisPhase = []
        for n in range(0,m):
            d = (2*n+1)/m
            if (m % 2) != 0:
                d1 = (2*n+1)/(m-1)
            else:
                d1 = (2*n+1)/m
            if d == 1:
                d = (4*n+1)/(2*m)
                divisor = math.sin(d*math.pi)
                mult = 1/divisor
                thisPhase.append(mult) # have to think about this
            else:
                divisor = math.sin(d1*math.pi)
                thisPhase.append(1/divisor)
        phases.append(thisPhase)

    #
    c = 2.99792E08  # m/s speed of light in vacuum
    distanceTraceInVacuum=round((len(Trace[0]) - hd["timeZero"])*hd["timeWindow"]/hd["numPoints"]*1E-9*c/2,2)
    samplesPerPeriodInVacuum = 1/((hd["freq"]*10**6) * 10**(-9))/(hd["timeWindow"]/hd["numPoints"])
    samplesPerTraceInVacuum = (len(Trace[0]) - hd["timeZero"])/samplesPerPeriodInVacuum
    
    maximumDepthX = -1
    maximumDepth = 0
    minimumDepthX = -1
    minimumDepth = 1000000 # arbitrary large number (max value is 65536)
    centuryDepth = [] # filled with triplets (avg,max,min) for 100 traces
    centuryAvg = 0
    centuryMax = 0
    centuryMin = 1000000
    
    print("")
    print(" vacuum distance",distanceTraceInVacuum,"samples per trace",samplesPerTraceInVacuum)
    print("")
    
    cycleNo = 0
    prevCycleNo = 0
    distanceTraceAverage = 0
    cycleLen = 0
    startIdx = 0
    cycleHist=np.zeros(int(maxCycleLen/2)+2, int)
    regionLen = (len(Trace[0])-int(hd["timeZero"]))/8
    cycleRegHist=np.zeros([8,int(maxCycleLen/2)+2], int)
    decadeHist=np.zeros((int(maxCycleLen/2)+2)*10, int)
    decadeRegHist=np.zeros([8,(int(maxCycleLen/2)+2)*10], int)
    zeroTime=int(hd["timeZero"]) - 7
    timeZero = int(hd["timeZero"])
    print(" start  stop   avg    max    min     delta")
    for k in range(0,len(Trace)-1):
        pos = neg = False
        cycleLen = 0
        startIdx = zeroTime
        center = 0
        parsedLongLine = False
        for i in range(zeroTime,len(Trace[0])-2):
            if neg == False and Trace[k][i] < 0:
                center = i
                neg = True
            if pos == False and Trace[k][i] >= 0:
                pos = True
            # add clarification for zeroes that are followed and proceeded by integers of same sign
            if (Trace[k][i] == 0) and ((Trace[k][i-1] < 0 and Trace[k][i+1] < 0) or (Trace[k][i-1] > 0 and Trace[k][i+1] > 0)):
                cycleLen += 1
                i += 1
                continue
            
            # look for continguous zeros
            zeroEnd = i
            if (Trace[k][i] == 0):
                while (Trace[k][zeroEnd+1] == 0 and (zeroEnd < len(Trace[0])-2)):
                    zeroEnd += 1

            if (pos == True and neg == True) and ((Trace[k][i] == 0) and (Trace[k][i-1] < 0 and Trace[k][zeroEnd+1] > 0)) or ((Trace[k][i] != 0) and (Trace[k][i] < 0 and Trace[k][i+1] > 0)):
                if (cycleLen <2):
                    continue
                if cycleNo != 0 and cycleLen > 1:
                    #if cycleLen > 19:
                    # Look for anomolies - those cycles that begin negative near timeZero
                    while Trace[k][startIdx] < 0 and startIdx < timeZero + 6:
                        startIdx += 1
                        cycleLen -= 1
                        if (center < startIdx): # the 'center' point was removed - so reset
                            for z in range(startIdx,i+1):
                                if (Trace[k][z] < 0):
                                    center = z 
                                    break
                    #
                    # Look for a more accurate period length
                    # Consider the start
                    #
                   
                    parsedLongLine = False
                    if Trace[k][startIdx] < 0 or cycleLen > 10: # look for anomolies
                        parsedLongLine = True
                        start = startIdx
                        maxima = minima = nextMaxima = 0
                        z = startIdx
                        while z < i+1:
                            # Break up duplicates
                            if (Trace[k][z] == Trace[k][z+1] and Trace[k][z] == Trace[k][z+2]):
                                endpoint = z + 2
                                while (Trace[k][z] == Trace[k][endpoint+1]):
                                    endpoint += 1
                                dupLen = endpoint-z+1
                                if (endpoint - z + 1 < 4):
                                    center = start
                                    while (Trace[k][center] > 0 and center < z+1):
                                        center += 1
                                    stop = z+1
                                    cycleFill(k,start,stop,center,maxCycleLen,phases,phaseData,invPhaseData,phaseDecadeData,cycleHist,cycleRegHist,decadeHist,decadeRegHist,"A")
                                    start = stop + 1
                                    maxima = minima = 0
                                    z += 1
                                    continue
                                else:
                                    center = start
                                    while (Trace[k][center] > 0 and center < z+1):
                                        center += 1
                                    if (z+1 - start < 2): # only one point but need two
                                        z += 1
                                    cycleFill(k,start,z,center,maxCycleLen,phases,phaseData,invPhaseData,phaseDecadeData,cycleHist,cycleRegHist,decadeHist,decadeRegHist,"B")
                                    center = z+1 + int((endpoint-(z+1))/2)
                                    cycleFill(k,z+1,endpoint,center,maxCycleLen,phases,phaseData,invPhaseData,phaseDecadeData,cycleHist,cycleRegHist,decadeHist,decadeRegHist,"C")
                                    start = endpoint+1
                                    maxima = minima = 0
                                    z = start
                                    continue
                            if (z == i):
                                if (z+1-start <= 10): # at 250 MHz and 40 ns - period of 10 in ice
                                    center = start
                                    while (Trace[k][center] > 0 and center < z+1):
                                        center += 1
                                    if (center == start): # only one point but need two
                                        z += 1
                                    cycleFill(k,start,z,center,maxCycleLen,phases,phaseData,invPhaseData,phaseDecadeData,cycleHist,cycleRegHist,decadeHist,decadeRegHist,"D")
                                    z += 1
                                else: # hot knife to force conformity to speed of light
                                    splitAt = start + int((z-1-start)/2)
                                    center = start
                                    while (Trace[k][center] > 0 and center < splitAt):
                                        center += 1
                                    cycleFill(k,start,splitAt,center,maxCycleLen,phases,phaseData,invPhaseData,phaseDecadeData,cycleHist,cycleRegHist,decadeHist,decadeRegHist,"E")
                                    center = splitAt
                                    while (Trace[k][center] > 0 and center < z+1):
                                        center += 1
                                    cycleFill(k,splitAt+1,z,center,maxCycleLen,phases,phaseData,invPhaseData,phaseDecadeData,cycleHist,cycleRegHist,decadeHist,decadeRegHist,"F")
                                    z += 1
                                continue
                            if (maxima == 0 and Trace[k][z] > Trace[k][z+1]):
                                # Check for one sample divergence
                                if (Trace[k][z] <= Trace[k][z+2]):
                                    z += 1
                                    continue
                                maxima = z
                                z += 1
                                continue
                            if (maxima > 0 and minima == 0 and Trace[k][z] < Trace[k][z+1]):
                                # Check for one sample divergence
                                if (Trace[k][z] >= Trace[k][z+2]):
                                    z += 1
                                    continue
                                minima = z
                                z += 1
                                continue
                            if (maxima > 0 and minima > 0):
                                # looking for the next maxima, the find the mid 
                                # point between minima and next maxima
                                if (Trace[k][z-1] < Trace[k][z] and Trace[k][z] > Trace[k][z+1]):
                                    # The next point was the next maxima so - yuck
                                    stop = z-1
                                    center = start
                                    while (Trace[k][center] >= 0 and center < stop+1):
                                        center += 1
                                    if (center == stop+1):
                                        center = start
                                        while (Trace[k][center] > 0 and center < stop+1):
                                            center += 1
                                        if (center == stop+1):
                                            center = stop
                                    cycleFill(k,start,z,center,maxCycleLen,phases,phaseData,invPhaseData,phaseDecadeData,cycleHist,cycleRegHist,decadeHist,decadeRegHist,"G")
                                    start = z
                                    maxima = z
                                    minima = 0
                                elif (Trace[k][z-1] < Trace[k][z] and Trace[k][z] < Trace[k][z+1] and Trace[k][z+1] >= Trace[k][z+2]):
                                    stop = minima + int((z+1-minima)/2)
                                    center = start
                                    while (Trace[k][center] >= 0 and center < stop+1):
                                        center += 1
                                    if (center == stop+1):
                                        center = start
                                        while (Trace[k][center] > 0 and center < stop+1):
                                            center += 1
                                        if (center == stop+1):
                                            center = stop
                                    cycleFill(k,start,stop,center,maxCycleLen,phases,phaseData,invPhaseData,phaseDecadeData,cycleHist,cycleRegHist,decadeHist,decadeRegHist,"H")
                                    start = stop + 1
                                    maxima = minima = 0
                            z += 1


                if parsedLongLine == False and (i+1 - startIdx > 1): # the second test occurs near zero line
                    cycleFill(k,startIdx,i+1,center,maxCycleLen,phases,phaseData,invPhaseData,phaseDecadeData,cycleHist,cycleRegHist,decadeHist,decadeRegHist,"I")
                cycleNo += 1
                cycleLen = 0
                startIdx = i+1
                neg = pos = False
            cycleLen += 1
        #  vacuum distance 136.27 samples per trace 227.273
        calcDist=round(distanceTraceInVacuum*(samplesPerTraceInVacuum/(cycleNo-prevCycleNo)),2)
        centuryAvg += calcDist
        if (calcDist > maximumDepth):
            maximumDepth = calcDist
            maximumDepthX = k
        if (calcDist > centuryMax):
            centuryMax = calcDist
        if (calcDist < minimumDepth):
            minimumDepth = calcDist
            minimumDepthX = k
        if (calcDist < centuryMin):
            centuryMin = calcDist
        if ((k+1)%100 == 0):
            print(" {0:5d} {1:5d}".format(k-99,k)," {0:6.2f} {1:6.2f} {2:6.2f}  {3:6.2f}".format(centuryAvg/100,centuryMax,centuryMin,centuryMax-centuryMin))
            centuryDepth.append((round(centuryAvg/100,2),centuryMax,centuryMin))
            centuryAvg = 0
            centuryMax = 0
            centuryMin = 1000000
            
        prevCycleNo = cycleNo

    print("  depth: max maxX  min minX",maximumDepth,"at",maximumDepthX," ",minimumDepth,"at",minimumDepthX)
    print("")
    sumCycle = 0
    maxPeriod = 0
    for i in range(len(cycleHist)):
        if (cycleHist[i] > 0):
            maxPeriod = i
        sumCycle += cycleHist[i] * i
    print(" Period CycleHist: number of periods", cycleNo," average period length:",round(sumCycle/cycleNo,2)," max",maxPeriod)
    print(cycleHist[0:maxPeriod+1])
    print("")
    for n in range(len(cycleRegHist)):
        sumCycle = 0
        numCycle = 0
        for i in range(len(cycleRegHist[n])):
            numCycle += cycleRegHist[n][i]
            sumCycle += cycleRegHist[n][i] * i
        refractiveIdx = 10*numCycle/sumCycle
        print("  reg ", n, " average: phase length", round(sumCycle/numCycle,2)," refractive Idx", round(refractiveIdx,2)," dielectric", round(refractiveIdx*refractiveIdx,2),"density",round((refractiveIdx-1)/0.845,3))
    for i in range(len(cycleHist)):
        cycleHist[i] *= i
    
    print("")
    print(" Points in CycleHist", cycleHist[0:maxPeriod+1])
    print("")
    print("   decadeHist   refractiveIdx    dielectric    density")
    guess = np.zeros(171,int)
    for n in range(len(decadeHist)):
        if (decadeHist[n] == 0) or (n == 0):
            continue
        refractiveIndex = 100/n  # 100 is 10 points x 10 (decase) which is the propagation of 250MHz in vaccuum
        dielectricConstant = refractiveIndex * refractiveIndex
        density = (refractiveIndex-1)/0.845
        #  Groupings should be <0.4, 0.4-0.83, >0.83 
        if (decadeHist[n] > 10000 or density > 0.4):
            print(" ",n,"   {0:6d}".format(decadeHist[n]),"     {0:5.2f}".format(100/n),"          {0:5.2f}".format(dielectricConstant),"     {0:6.3f}".format(density))
            guess[n] = decadeHist[n]
     
    # 
    #
    #
    #
    c = 2.99792E08  # m/s speed of light in vacuum
    distanceTraceInVacuum=round((len(Trace[0]) - hd["timeZero"])*hd["timeWindow"]/hd["numPoints"]*1E-9*c/2,2)
    samplesPerPeriodInVacuum = 1/((hd["freq"]*10**6) * 10**(-9))/(hd["timeWindow"]/hd["numPoints"])
    samplesPerTraceInVacuum = (len(Trace[0]) - hd["timeZero"])/samplesPerPeriodInVacuum
    print("")
    print(" vacuum distance",distanceTraceInVacuum,"samples per trace",samplesPerTraceInVacuum)
    print("")
    
    #hd["numPoints"]  # number of points sampled in time Window
    #hd["timeWindow"] # in nanoseconds
    #hd["timeZero"]   # e.g. 227 for dt10
    #hd["freq"]       # in MHz
    c = 2.99792E08  # m/s speed of light in vacuum
    ## Get distance of entire trace in samples(pixels)
    distanceSample = len(Trace[0]) - hd["timeZero"]
    ## Find the period of the nominal frequency
    period = 1/((hd["freq"]*10**6) * 10**(-9)) # to get ns
    ## number of samples needed to read one period
    timePerSample = hd["timeWindow"]/hd["numPoints"] # per nanosecond
    samplesPerPeriodInVacuum = period/timePerSample

    print(" In Vacuum: period",period,"ns time per sample",timePerSample,"ns therefore samples per period:",samplesPerPeriodInVacuum)
    # Find average number of samples per period for the Trace
    numSamples = round(len(Trace)*distanceSample)
    periodsPerTraceInVacuum = round(numSamples/samplesPerPeriodInVacuum)
    print(" numSamples",numSamples," periodsPerTraceInVacuum",periodsPerTraceInVacuum," Total periods found",cycleNo)
    print("  depth,two way, in vacuum", round((distanceSample*timePerSample)))
    print(" As waves enter more dense material, the frequency remains the same but the velocity slows and the period increases")
    ratio = round(cycleNo/periodsPerTraceInVacuum,3)
    print(" ratio:", ratio)
    distanceTraceInVacuum=round(distanceSample*timePerSample*1E-9*c/2,2) 
    distanceTraceRuleOfThumb=round(distanceTraceInVacuum*2/3,2)
    distanceTraceAverage=round(distanceTraceInVacuum/ratio,2)
    print(" TwTt  distance traveled in meters: vacuum",distanceTraceInVacuum,"ruleOfThumb",distanceTraceRuleOfThumb,"calc",distanceTraceAverage)
    """    
    # Five wavelength moving average
    iceLine = copy.deepcopy(phaseDecadeData)
    yMax = len(iceLine[0])
    iceThreshold = 66
    numWaves = 3
    max = 0
    min= 32000

    for x in range(len(iceLine)):
        y = 0
        strength = 0
        length = 0
        waves = []
        while y < yMax:
            if (iceLine[x][y] == 0):
                y += 1
                continue
            if (len(waves) < numWaves):
                waveLength = int(iceLine[x][y]/10)
                waveStr = iceLine[x][y] * waveLength
                length += waveLength
                strength += waveStr
                waves.append((y,waveLength,waveStr))
                y += waveLength
                if (len(waves) == numWaves):
                    targetY,targetLen,targetStr = waves.pop(0)
                    avgStrength = int(strength/length)
                    if (avgStrength > max):
                        max = avgStrength
                    elif (avgStrength < min):
                        min = avgStrength
                    for z in range(targetY,targetY+targetLen+1):
                        iceLine[x][z] = avgStrength
                    strength -= targetStr
                    length -= targetLen
   
    graphData(iceLine, "iceLine - moving average", 'gist_rainbow', 0 , False)
    
       
    #  wavelength moving average
    #iceAvg = copy.deepcopy(phaseDecadeData)
    iceAvg = np.zeros([len(Trace), len(Trace[0])], int)
    yMax = len(iceAvg[0])
    numSamples = 5
    numSquare = numSamples * numSamples
    numAdj = int(numSamples/2)
    minG = 65565
    maxG = 0
    histIce = [0] * 20
    
 
    for x in range(numAdj,len(iceAvg)-numAdj):
        for y in range(numAdj,len(iceAvg[0])-numAdj):
            if (phaseDecadeData[x][y] == 0):
                y += 1
                histIce[0] += 1
                continue
            sumIce = 0
            realSamples = 0
            for m in range(x-numAdj, x+numAdj):
                for n in range(y-numAdj, y+numAdj):
                    if (phaseDecadeData[m][n] == 0):
                        continue
                    sumIce += phaseDecadeData[m][n]
                    realSamples += 1
            if (realSamples != 0):
                value = float(sumIce/realSamples)
                gValue = int(value/10)
                if (gValue < 5):
                    gValue = 1
                else:
                    gValue -= 4
                    if (gValue > 7):
                        gValue = 7
                if (gValue < len(histIce)):
                    histIce[gValue] += 1
                else:
                    print(" gValue",gValue," greater than ", len(histIce))
                iceAvg[x][y] = gValue
                #if (value < minG):
                #    minG = value
                #if (value > maxG):
                #    maxG = value
            else:
                histIce[0] += 1
    print("  histIce",histIce)
   
    graphData(iceAvg, "iceAvg", 'Accent', 0 , False)
    
   
    #sys.exit(0)
    print("")

    """    
    #  Can't Sleep
    cantSleep = np.full([len(Trace), len(Trace[0])], 5, int)
    length = 13
    midPoint = int(length/2) 
    
    print("This takes a while - averageing points many times so x will be currently printed")
   
 
    for x in range(midPoint,len(cantSleep)-midPoint):
        if (x%50 == 0):
            print(" x ",x)
        for y in range(midPoint+int(hd["timeZero"]),len(cantSleep[0])-midPoint):
            sumIce = 0
            realSamples = 0
            for m in range(x-midPoint, x+midPoint):
                for n in range(y-midPoint, y+midPoint):
                    #if (phaseDecadeData[m][n] == 0):
                    #    continue
                    sumIce += phaseDecadeData[m][n]
                    realSamples += 1
            if (realSamples != 0):
                value = float(sumIce/realSamples)
                #print("x,y",x,y,"value",value,"sum",sumIce,"n",realSamples)
                decPeriod = 5
                if (value < 67.0):
                    decPeriod = 0
                elif (value < 70.0):
                    decPeriod = 1
                elif (value < 73.0):
                    decPeriod = 3
                else:
                    decPeriod = 4                
                cantSleep[x][y] = decPeriod
                if (x == midPoint):
                    for z in range(0,midPoint):
                        cantSleep[z][y] = decPeriod
                elif (x == len(cantSleep)-midPoint-1):
                    for z in range(len(cantSleep)-midPoint-1,len(cantSleep)-1):
                        cantSleep[z][y] = decPeriod

   
    graphDataColour1(cantSleep, "2018 line10 - Averaged Periods", 'terrain')
    
    print("")
    """
    Colours=('blue','green','red','cyan','magenta','yellow','black','blue','green','red','cyan','magenta','yellow','black','blue','green','red','cyan','magenta','yellow','black',)

    plt.title('Periods Vs Region', fontsize=14)
    plt.xlabel('Periods', fontsize=14)
    plt.ylabel('Region', fontsize=14)
    plt.grid(True)
    validPeriods = []
    for n in range(len(cycleHist)-1):
        if (cycleHist[n] > 10000):
            validPeriods.append(n)
    print("  validPeriods greater than 10000 of ",cycleNo,":",validPeriods)
    
    region = []    
    for m in range(len(cycleRegHist)-1):
        for n in validPeriods:
            region.append(cycleRegHist[m][n])
            #print(" ",m," n",n," ",cycleRegHist[m][n],"region",region)
        print("  reg ", m, " region", region, " entire[0:10]",cycleRegHist[m][0:10])
        plt.plot(validPeriods, region, color=Colours[m], marker='o')
        region.clear()
    plt.show()
    """             
    
    """
    Colours=('blue','green','red','cyan','magenta','yellow','black','tab:orange','tab:brown','blue','green','red','cyan','magenta','yellow','black','blue','green','red','cyan','magenta','yellow','black',)

    plt.title('Periods X 10 Vs Region', fontsize=14)
    plt.xlabel('Periods X 10', fontsize=14)
    plt.ylabel('Region', fontsize=14)
    plt.grid(True)
    plt.figure(figsize=(32,10))
    validPeriods = []
    for n in range(len(decadeHist)-1):
        if (decadeHist[n] > 1000):
            validPeriods.append(n)
    print("  valid Period*10 greater than 1000 of ",cycleNo,":",validPeriods)
    
    region = []    
    for m in range(len(decadeRegHist)):
        for n in validPeriods:
            region.append(decadeRegHist[m][n])
            #print(" ",m," n",n," ",cycleRegHist[m][n],"region",region)
        print("  reg ", m, " region", region, " entire[0:10]",decadeRegHist[m][0:10])
        plt.plot(validPeriods, region, color=Colours[m], marker='o')
        region.clear()
    for n in validPeriods:
        region.append(decadeHist[n]/4)
    plt.plot(validPeriods, region, 'tab:pink', marker='+')
    region.clear()
    plt.show()
    
    sys.exit(0)
    """  
    #traceTPData = np.zeros((len(Trace),len(Trace[0])))
    #for j in range(len(Trace)):
    #    for k in range(len(Trace[0])):
    #        traceTPData[j,k]= invPhaseData[j][k]

    #vbpTrace = verticalBandPass(hd,traceTPData,15,450)
    #remAvgVbpInvPhaseData = removeAverage(vbpTrace, maxValue,1)
    #graphData(remAvgVbpInvPhaseData,"  invPhase, vbp 15-450, har", 'YlGn', 0 , False)
    
    #remAvgInvPhaseData = removeAverage(invPhaseData, maxValue,1)
    #harPhaseTrace = removeAverage(phaseTrace, maxValue,1)
    #graphData(harPhaseTrace, "harPhaseData YlGn", 'YlGn', 0 , False)
    #graphData(harPhaseTrace, "harPhaseData hsv", 'hsv', 0 , False)
    
    #testTrace = np.zeros([len(phaseData),len(phaseData[0])],int)
    #for i in range(len(phaseData)):
    #    for j in range(len(phaseData[0])):
    #        if phaseData[i][j] == 8 or phaseData[i][j] == 9 or phaseData[i][j] == 13:
    #            testTrace[i][j] = phaseData[i][j]
    #graphData(testTrace, "test 8,9,13", 'Accent', 0 , False)
    """
    testTrace = np.zeros([len(phaseData),len(phaseData[0])],int)
    for i in range(len(phaseData)):
        for j in range(len(phaseData[0])):
            if phaseData[i][j] == 5:
                testTrace[i][j] = phaseData[i][j]
    graphData(testTrace, "test 5", 'Accent', 0 , False)
    
       
    testTrace = np.zeros([len(phaseData),len(phaseData[0])],int)
    for i in range(len(phaseData)):
        for j in range(len(phaseData[0])):
            if phaseData[i][j] == 6:
                testTrace[i][j] = phaseData[i][j]
    graphData(testTrace, "test 6", 'Accent', 0 , False)
    
    testTrace = np.zeros([len(phaseData),len(phaseData[0])],int)
    for i in range(len(phaseData)):
        for j in range(len(phaseData[0])):
            if phaseData[i][j] == 7:
                testTrace[i][j] = phaseData[i][j]
    graphData(testTrace, "test 7", 'Accent', 0 , False)
       
    testTrace = np.zeros([len(phaseData),len(phaseData[0])],int)
    for i in range(len(phaseData)):
        for j in range(len(phaseData[0])):
            if phaseData[i][j] == 8:
                testTrace[i][j] = phaseData[i][j]
    graphData(testTrace, "test 8", 'Accent', 0 , False)        
    """
    testTrace = np.zeros([len(phaseData),len(phaseData[0])],int)
    for i in range(len(phaseData)):
        for j in range(len(phaseData[0])):
            if phaseData[i][j] == 7 or phaseData[i][j] == 8:
                testTrace[i][j] = phaseData[i][j]
    graphData(testTrace, "test 7,8", 'tab10', 0 , False)
    """  
    testTrace = np.zeros([len(phaseData),len(phaseData[0])],int)
    for i in range(len(phaseData)):
        for j in range(len(phaseData[0])):
            if phaseData[i][j] == 9:
                testTrace[i][j] = phaseData[i][j]
    graphData(testTrace, "test 9", 'Accent', 0 , False)
       
    testTrace = np.zeros([len(phaseData),len(phaseData[0])],int)
    for i in range(len(phaseData)):
        for j in range(len(phaseData[0])):
            if phaseData[i][j] == 10:
                testTrace[i][j] = phaseData[i][j]
    graphData(testTrace, "test 10", 'Accent', 0 , False)
        
    testTrace = np.zeros([len(phaseData),len(phaseData[0])],int)
    for i in range(len(phaseData)):
        for j in range(len(phaseData[0])):
            if phaseData[i][j] == 11:
                testTrace[i][j] = phaseData[i][j]
    graphData(testTrace, "test 11", 'Accent', 0 , False)
       
      
    testTrace = np.zeros([len(phaseData),len(phaseData[0])],int)
    for i in range(len(phaseData)):
        for j in range(len(phaseData[0])):
            if phaseData[i][j] == 12:
                testTrace[i][j] = phaseData[i][j]
    graphData(testTrace, "test 12", 'Accent', 0 , False)
    
    testTrace = np.zeros([len(phaseData),len(phaseData[0])],int)
    for i in range(len(phaseData)):
        for j in range(len(phaseData[0])):
            if phaseData[i][j] == 13:
                testTrace[i][j] = phaseData[i][j]
    graphData(testTrace, "test 13", 'Accent', 0 , False)
       
    testTrace = np.zeros([len(phaseData),len(phaseData[0])],int)
    for i in range(len(phaseData)):
        for j in range(len(phaseData[0])):
            if phaseData[i][j] == 14:
                testTrace[i][j] = phaseData[i][j]
    graphData(testTrace, "test 14", 'Accent', 0 , False)        
    
    testTrace = np.zeros([len(phaseData),len(phaseData[0])],int)
    for i in range(len(phaseData)):
        for j in range(len(phaseData[0])):
            if phaseData[i][j] == 15:
                   testTrace[i][j] = phaseData[i][j]
    graphData(testTrace, "test 15", 'Accent', 0 , False)
       
    testTrace = np.zeros([len(phaseData),len(phaseData[0])],int)
    for i in range(len(phaseData)):
        for j in range(len(phaseData[0])):
            if phaseData[i][j] == 16:
                   testTrace[i][j] = phaseData[i][j]
    graphData(testTrace, "test 16", 'Accent', 0 , False)
    """
  
    """                 
    testTrace = np.zeros([len(phaseData),len(phaseData[0])],int)
    for i in range(len(phaseData)):
        for j in range(len(phaseData[0])):
            if phaseData[i][j] == 2 or phaseData[i][j] == 3 or phaseData[i][j] == 16:
                   testTrace[i][j] = phaseData[i][j]
    graphData(testTrace, "test 2,3,16", 'Accent', 0 , False)
                 
    testTrace = np.zeros([len(phaseData),len(phaseData[0])],int)
    for i in range(len(phaseData)):
        for j in range(len(phaseData[0])):
            if phaseData[i][j] == 4 or phaseData[i][j] == 5 or phaseData[i][j] == 15:
                   testTrace[i][j] = phaseData[i][j]
    graphData(testTrace, "test 4,5,15", 'Accent', 0 , False)
                  
    
    testTrace = np.zeros([len(phaseData),len(phaseData[0])],int)
    for i in range(len(phaseData)):
        for j in range(len(phaseData[0])):
            if phaseData[i][j] == 6 or phaseData[i][j] == 7 or phaseData[i][j] == 14:
                   testTrace[i][j] = phaseData[i][j]
    graphData(testTrace, "test 6,7,14", 'Accent', 0 , False)
                 
    testTrace = np.zeros([len(phaseData),len(phaseData[0])],int)
    for i in range(len(phaseData)):
        for j in range(len(phaseData[0])):
            if phaseData[i][j] == 8 or phaseData[i][j] == 9 or phaseData[i][j] == 13:
                   testTrace[i][j] = phaseData[i][j]
    graphData(testTrace, "test 8,9,13", 'Accent', 0 , False)
  
                 
    testTrace = np.zeros([len(phaseData),len(phaseData[0])],int)
    for i in range(len(phaseData)):
        for j in range(len(phaseData[0])):
            if phaseData[i][j] == 10 or phaseData[i][j] == 11 or phaseData[i][j] == 12:
                   testTrace[i][j] = phaseData[i][j]
    graphData(testTrace, "test 10,11,12", 'Accent', 0 , False)
    """                     
    #"""
    graphData(phaseData, "phaseData", 'tab20c', 0 , False)
    graphData(phaseData, "phaseData hsv", 'hsv', 0 , False)
    graphData(phaseDecadeData, "phaseDecadeData hsv", 'hsv', 0 , False)
    #"""
   
    #graphData(invPhaseData, "invPhaseData", 'YlGn' , 0 , False)
    #graphData(remAvgInvPhaseData, "remAvgInvPhaseData", 'YlGn', 0 , False)
    
    return(phaseData,invPhaseData)

        
def parabola(allLines, Trace, threshold, bandLength=50, style=0):
    """ Approximate truncated parabolas. 

    Parameters:
        allLines: list of list of lines 
        Trace: list of list of backscatter/strengths
        threshold: the y limit of the bedrock
        bandLength: minimum length for a parabola
        style: 0=pure polyfit, 1=polyfit to start-go up to find terminus

    Function:
        Large parabolas can be truncated at threshold of Analyse, so use polyfit to estimate the missing top and if style=1 use vertical backscatter to find upward limit. 

    Return Values:
        none
    """
    print("parabola: threshold",threshold,"bandLength",bandLength,"style",style)
    #
    # style: 0=pure polyfit, 1=polyfit to start-go up to find terminus
    #
    # Calculate the unknowns of the equation y=ax^2+bx+c via numpy polyfit 
    #
    # Find the first line of appropriate length
    #
    # returns the pinch point matrix
    targetLine =0
    for x in range(len(allLines)):
        if (len(allLines[x]) >= bandLength):
            targetLine = x
            break
    print(" Choose target line",targetLine)
    # np.polyfit setup
    #
    # Done as fixed array as I can't get np.append(x,number) working
    #
    arrayLen = 9
    x = np.zeros(arrayLen) 
    y = np.zeros(arrayLen)
    idx = 0
    x[idx] = allLines[targetLine][0][3]
    y[idx] = allLines[targetLine][0][0] - allLines[targetLine][0][1]
    idx += 1
    xi = xf = 0
    Xstart = 0
    
    # Left Nadir point is the first point that exceeds: threshold - 100
    lineLen = len(allLines[targetLine])
    limit = threshold + 200
    a = 1
    while (a < lineLen):
        yN = allLines[targetLine][a][0] - allLines[targetLine][a][1]
        if (yN < limit):
            xi = allLines[targetLine][a][3]
            x[idx] = xi
            y[idx] = yN
            idx += 1
            Xstart = a
            break
        if (a % 10 == 0) and (idx < (arrayLen - 2)):
            x[idx] = allLines[targetLine][a][3]
            y[idx] = yN
            idx += 1
        a += 1
    if (a >= lineLen):
        print(" targetLine, a", targetLine, a, "exceeded lineLLen", lineLen)
        return [0, 0]
    xLeft = allLines[targetLine][a][3]
    # Last point is the Right nadir
    a = lineLen - 1
    while (a > 0):
        yN = allLines[targetLine][a][0] - allLines[targetLine][a][1]
        if (yN < limit):
            xf = allLines[targetLine][a][3]
            x[idx] = xf
            y[idx] = yN
            idx += 1
            break
        a -= 1
    xRight = allLines[targetLine][a][3]
    #
    # Calculate polyfit
    #
    coEff = np.polyfit(x,y,2)
    #
    # Use polyfit to find vertex
    #  
    for x in range(xi, xf, 1):
        y = int(round((coEff[0]*(x**2))+(coEff[1]*x)+coEff[2]))
        if (style == 1):
            yi = y
            y -= 1
            while (abs(Trace[x][y]) > 1):
                y -= 1
        depth = allLines[targetLine][Xstart][1]
        allLines[targetLine][Xstart][1] = allLines[targetLine][Xstart][0] - y
        Xstart += 1
    pinchPoint = [xLeft, xRight]

    return pinchPoint

def bottomDensity(Trace, threshold):
    """ Calculate the bedrock backscatter strength. 

    Parameters:
        Trace: list of list of strengths/backscatter
        threshold: the density of the Band for the bottom

    Function:
        Calculate the density of the bottom parabola.

    Return Values:
        density
    """
    totalTraces = 0
    totalStrength = 0
    for i in range(len(Trace)):
        for j in range(threshold,len(Trace[0])-1):
            totalStrength += Trace[i][j]
            totalTraces += 1
    density = int(totalStrength/totalTraces)
    return density


def harVar(Trace, threshold=100, zeroPoint=1):
    """ Vertical average followed by horizontal averaging. 

    Parameters:
        Trace: list of list of line strengths/backscatter 
        threshold: the maximum value of the remaining values
        zeroPoint: the y coordinate of the zero gap 

    Function:
        Remove both vertical and horizontal average. This routine was developed for very noisy columnar banding.

    Return Values:
        none
    """
    print("harVar: remove average from both row and column, threshold", threshold, "zeroPoint",zeroPoint)

    #
    # Remove the average from each point on line, assumes absolute values
    #
    newTrace = copy.deepcopy(Trace)
    
    xRange = len(newTrace)
    yRange = len(newTrace[0])
    harRemoved = 0
    varRemoved = 0
    total = 0
    lineAvgs = []

    for x in range(xRange-1):
        total = 0
        for y in range(0, yRange-1):
            if (newTrace[x][y] < 0):
                newTrace[x][y] *= -1 # make absolute
            newTrace[x][y] += abs(Trace[x][y] - Trace[x][y-1]) # add derivative
            if (y > zeroPoint):
                total += newTrace[x][y]

        average = round(2*total/(3*(yRange-zeroPoint)))# + 1
        lineAvgs.append(average)
        # This is evil - attemptin to reverse curse
        # average = 48

        threshold = 1000

        for y in range(yRange-1):
            if (newTrace[x][y] < average):
                newTrace[x][y] = 0
                varRemoved += 1
            else:
                if (newTrace[x][y] > threshold):
                    newTrace[x][y] = threshold
   
    for y in range(yRange-1):
        total = 0
        for x in range(xRange-1):
            total += newTrace[x][y]

        average = round(2*total/(3*xRange))# + 1
        for x in range(xRange-1):
            if (newTrace[x][y] < average):
                newTrace[x][y] = 0
                harRemoved += 1
            else:
                newTrace[x][y] -= average
                if (newTrace[x][y] > threshold):
                    newTrace[x][y] = threshold
    print(" average removed: var", varRemoved, " har", harRemoved)                

    return newTrace

def varAverage(Trace, threshold=100, zeroPoint=1):
    """ Vertical average removal. 

    Parameters:
        Trace: list of list of line strengths/backscatter 
        threshold: the maximum value of the remaining values
        zeroPoint: the y coordinate of the zero gap 

    Function:
        Remove the vertical average. This routine was necessary for very noisy columnar banding.

    Return Values:
        averaged Trace
    """
    print("varAverage: remove average from each column, threshold", threshold, "zeroPoint",zeroPoint)
    #
    # Remove the average from each point on line, assumes absolute values
    #
    newTrace = copy.deepcopy(Trace)
    
    xRange = len(newTrace)
    yRange = len(newTrace[0])
    removed = 0
    total = 0
    lineAvgs = []

    for x in range(xRange-1):
        total = 0
        for y in range(0, yRange-1):
            if (newTrace[x][y] < 0):
                newTrace[x][y] *= -1 # make absolute
            newTrace[x][y] += abs(Trace[x][y] - Trace[x][y-1]) # add derivative
            if (y > zeroPoint):
                total += newTrace[x][y]

        average = round(total/(yRange-zeroPoint)) + 1
        lineAvgs.append(average)

        threshold = 1000

        for y in range(yRange-1):
            if (newTrace[x][y] < average):
                newTrace[x][y] = 0
                removed += 1
            else:
                #newTrace[x][y] -= average
                if (newTrace[x][y] > threshold):
                    newTrace[x][y] = threshold
             
    print(" average removed: ", removed)
    return newTrace


def varAverage2(Trace, maxValue=10, zeroPoint=1):
    """ Remove average from lowest strength column. 

    Parameters:
        Trace: list of list of line strengths/backscatter 
        maxValue: the maximum value of the remaining values
        zeroPoint: the y coordinate of the zero gap 

    Function:
        Remove the average from the lowest column each point on line, assumes absolute values

    Return Values:
        averaged Trace
    """
    print("varAverage2: remove average from lowest strength column, maxValue", maxValue, "zeroPoint",zeroPoint)
    #
    # Remove the average from the lowest column each point on line, assumes absolute values
    #
    if (zeroPoint <= 1):
        print(" zeroPoint",zeroPoint,"too low - aborting")
    newTrace = copy.deepcopy(Trace)
    
    xRange = len(newTrace)
    yRange = len(newTrace[0])
    removed = 0
    lineHighTop = []
    lineHighBottom = []

    # Find the highest column average
    for x in range(xRange-1):
        high = 0
        for y in range(zeroPoint-1):
            if (newTrace[x][y] < 0):
                newTrace[x][y] *= -1 # make absolute
            if (int(newTrace[x][y]) > high):
                high = int(newTrace[x][y])
        lineHighTop.append(high)
        high = 0
        threshold = yRange - int((yRange-zeroPoint)*0.4)
        for y in range(threshold, yRange-1):
            if (newTrace[x][y] < 0):
                newTrace[x][y] *= -1 # make absolute
            if (int(newTrace[x][y]) > high):
                high = int(newTrace[x][y])
        lineHighBottom.append(high)

    # Remove the highest found before the zeroPoint
    
    print("")
    print ("  lineHighTop", lineHighTop)
    print("")
    print ("  lineHighBottom", lineHighBottom)

    for x in range(xRange-1):
        for y in range(yRange-1):
            if (newTrace[x][y] < int(4*lineHighBottom[x]/5)):
                newTrace[x][y] = 0
                removed += 1
            else:
                newTrace[x][y] -= int(4*lineHighBottom[x]/5)
                """
                # logarithmic value
                # 4,16,64,128,512,2048,8192, greater
                # 1, 2, 3,  4,  5,   6,   7,    8

                if (newTrace[x][y] > 8192):
                    newTrace[x][y] = 10
                elif (newTrace[x][y] > 2048):
                    newTrace[x][y] = 10
                elif (newTrace[x][y] > 1024):
                    newTrace[x][y] = 9
                elif (newTrace[x][y] > 512):
                    newTrace[x][y] = 9
                elif (newTrace[x][y] > 256):
                    newTrace[x][y] = 8
                elif (newTrace[x][y] > 128):
                    newTrace[x][y] = 7
                elif (newTrace[x][y] > 64):
                    newTrace[x][y] = 6
                elif (newTrace[x][y] > 32):
                    newTrace[x][y] = 6
                elif (newTrace[x][y] > 16):
                    newTrace[x][y] = 5
                else:
                    newTrace[x][y] = 3
                #if (newTrace[x][y] > maxValue):
                #    newTrace[x][y] = maxValue
                """
            
    print(" var average removed: ", removed)
    
    return newTrace

def signalToNoiseRatio(Trace,allLines,lineLen,yIdx,start,stop,interval,threshold=210):
    """ Denote the signal strength of the line compared to off-line. 

    Parameters:
        Trace: list of lists of strength/backscatter 
        allLines: list of lists of lines
        lineLen: minimum line length
        yIdx: isolate region of desired line
        start: starting x
        stop: stopping x
        interval: number of points to average
        threshold: strength

    Function:
        Follow a line and report signal strength within the line and above the line.

    Return Values:
        none
    """

    print("signalToNoiseRatio:  lineLen",lineLen,"yIndex",yIdx,"start",start,"stop",stop,"interval",interval)

    lineNo = -1
    for z in range(len(allLines)):
        if (len(allLines[z]) == 0):
            continue
        if (len(allLines[z]) >= lineLen):
            if (allLines[z][0][0] < yIdx):
                lineNo = z

    print(" lineNo", lineNo,"selected. len",len(allLines[lineNo]))
    if (lineNo < 0):
        print(" signalToNoiseRatio: no line found")
        return
    yLimit = len(Trace[0])-1
    xLimit = len(Trace)-1
    idx = 0
    if (allLines[lineNo][0][3] > 0):
        idx = allLines[lineNo][0][3]
    thresholdDetected = False
    if (threshold > 1000):
        thresholdDetected = True
        print("  Threshold detected:", threshold)


    for i in range(start-idx,stop-idx,interval):
        signal = noise = rand = 0
        for z in range(interval):
            if (i+z > len(allLines[lineNo])-1):
                break
            thickness = allLines[lineNo][i+z][1]
            scanThickness = thickness
            if (thickness > 50): # if examining bedrock, the thickness exceed half the height of the Trace
                scanThickness = 4
            plankOffset = 3 # plank thickness adjustment
            y = allLines[lineNo][i+z][0]
            yt = allLines[lineNo][i+z][0] - allLines[lineNo][i+z][1]                      
            x = allLines[lineNo][i+z][3]
            # make the area like the plank: thickness to width -3 to +3
            aboveThreshold = False
            for j in range(-2,scanThickness+2):
                for q in range(-1*plankOffset,plankOffset):
                    if (i+z+q > xLimit):
                        continue
                    if (i+z+q < 0):
                        continue
                    yOffset = y-thickness+j
                    if (yOffset > yLimit):
                        yOffset = yLimit
                    if (yt-j-plankOffset < threshold):
                        aboveThreshold = True
                        continue
                    signal += abs(Trace[i+z+q][yOffset])
                    yOffset = yt-j-2*plankOffset
                    if (yOffset > yLimit):
                        yOffset = yLimit
                    noise  += abs(Trace[i+z+q][yOffset])
                    yOffset = yt+j+plankOffset
                    if (yOffset > yLimit):
                        yOffset = yLimit-scanThickness+q
                        if (yOffset > yLimit):
                            yOffset = yLimit
                    rand += abs(Trace[i+z+q][yOffset])
        if noise == 0:
            print(" i",i,"x",x,"yt",yt,"signal",signal,"noise",noise,"rand",rand,"s/n: N/A n/n N/A thickness",thickness)
        else:
            if thresholdDetected == False:
                print(" I",i,"x",x,"yt",yt,"signal",signal,"noise",noise,"rand",rand,"s/n:{:.3f}".format(signal/noise),"n/n:{:.3f}".format(rand/noise),"diff:{:5.1f}".format(((signal/noise-1)*100)),"thickness",thickness)
            else:
                print(" i",i,"x",x,"yt",yt,"signal",noise,"noise",signal,"rand",rand,"s/n:{:.3f}".format(noise/signal),"n/n:{:.3f}".format(rand/noise),"diff:{:5.1f}".format(((noise/signal-1)*100)),"thickness",thickness)
    return
 

#
# Main routine
#
#
beginTime = datetime.datetime.now()

#binary_file =  "C:/temp/yyline3.DT1"
#binary_file =  "C:/temp/line09.DT1"
#text_file = "C:/temp/line09.HD"
#binary_file =  "C:/temp/line10.DT1"
#text_file = "C:/temp/line10.HD"
#binary_file =  "C:/temp/line11.DT1"
#text_file = "C:/temp/line11.HD"
#binary_file =  "C:/temp/line11_2017.DT1"
#text_file = "C:/temp/line11_2017.HD"
#binary_file =  "C:/temp/line12.DT1"
#text_file = "C:/temp/line12.HD"


# Read the .HD file
hd,twtt = parseHD(text_file)

# Read the .DT1 file
Trace = parseDTn(binary_file)

#
# Remove duplicates from the original Trace
#
dupRmvTrace = copy.deepcopy(Trace)
windowTrace = window(Trace, 1, 200, len(Trace)-1, 500, Data=False)
OddityFlag = False   # Can't run some subroutines on LINE11_2017 - hence Oddity Flag
remove_list = detectDuplicates(windowTrace, 20)
if (len(remove_list) > 50):
    OddityFlag = True
    print("")
    print(" Oddity detected - remove length", len(remove_list)," of",len(Trace))
    print("")
    maxValue = 10 # value used to graph data by minimizing the display range of values 
else:
    removeDuplicates(dupRmvTrace, remove_list)
    maxValue = 20 # value used to graph data by minimizing the display range of values
    
harTrace = removeAverage(dupRmvTrace, maxValue, 0) # 0 indicates to return a signed Trace
graphData(harTrace, "Horizontal Remove Average - seismic", 'seismic', 0 , False)
graphData(harTrace, "Horizontal Remove Average - YlGn", 'YlGn', 0 , False)


if OddityFlag == True:  # Can't run vbp or unWave on LINE11_2017
    varTrace = varAverage(dupRmvTrace, 20, 1400)
    harVarTrace = harVar(dupRmvTrace, 3, 1400) # maxValue = 100
    #graphData(varTrace, "varTrace", 'YlGn', 0 , False)
    #graphData(harVarTrace, "harVar", 'YlGn', 0 , False)
    #phaseTrace, invPhaseTrace = unWave(varTrace, maxValue)
    #harInvPhaseTrace = removeAverage(invPhaseTrace, maxValue, 1)
    #graphData(harInvPhaseTrace, "harInvPhaseTrace", 'YlGn', 0 , False)
    #harVarTrace = removeAverage(varTrace, maxValue, 1)
    #graphData(harVarTrace, "harVar", 'YlGn', 0 , False)
else:
    traceTPData = np.zeros((len(dupRmvTrace),len(dupRmvTrace[0])))
    for j in range(len(dupRmvTrace)):
        for k in range(len(dupRmvTrace[0])):
            traceTPData[j,k]= dupRmvTrace[j][k]
    xData = copy.deepcopy(traceTPData)

    vbpTrace = verticalBandPass(hd,xData,15,450)
    vbpHarTrace = removeAverage(vbpTrace, maxValue, 1)
    #graphData(vbpHarTrace, " vbp 15-450, har", 'seismic', 0 , False)
    graphData(vbpHarTrace, " vbp 15-450, har", 'YlGn', 0 , False)

    unWaveStartTime = datetime.datetime.now()
    phaseTrace, invPhaseTrace = unWave(dupRmvTrace, maxValue)
    unWaveEndTime = datetime.datetime.now()
    print("timex unWave",unWaveEndTime-unWaveStartTime)
    harInvPhaseTrace = removeAverage(invPhaseTrace, maxValue, 1)
    graphData(harInvPhaseTrace, "harInvPhaseTrace", 'YlGn', 0 , False)
#sys.exit(0)

#
# Filter selection: invPhase, dupRmv, vbp
#
if OddityFlag == True:  # Can't run vbp on LINE11_2017
    sampleTrace = varTrace
    TraceType   = "var"
    #sampleTrace = dupRmvTrace
    #TraceType   = "dupRmv"
else:
    sampleTrace = invPhaseTrace # not an Oddity option
    TraceType   = "invPhase"
    #sampleTrace = dupRmvTrace
    #TraceType   = "dupRmv"
    #sampleTrace = vbpTrace
    #TraceType   = "vbp"
    
#
# Threshold value selections
#

depth=5  # the depth of the band
width=7  # the width of the band
maxNumLines = (int)(len(sampleTrace)*0.10)   # arbitrary limit
minLineLength = (int)(len(sampleTrace)*0.01) # arbitrary limit
print(" maxNumLines",maxNumLines,"minLineLength",minLineLength)
nBands=maxNumLines
nData=7 #y,depth,energy,x,used,avg5depth,avg5midpoint

if (OddityFlag == False):
    ThinLines=True
    bandLength=30
else:
    ThinLines=False
    bandLength=500

bRNumLines = 701              # number of lines to process 
if (TraceType == "invPhase"):
    bedRockThreshold = 1450   # y co-ordinate above which the signal is too difuse
    bRStrengthThreshold = 17  #
elif (TraceType == "vbp"):
    bedRockThreshold = 1450   
    bRStrengthThreshold = 8   # 8,7 shows full 1st bedrock but dwarf second,
elif (TraceType == 'var'):    # Oddity
    bedRockThreshold = 4000   
    bRStrengthThreshold = 700 
    bRNumLines = 21
    ThinLines = False
else: # default dupRmvTrace
    bedRockThreshold = 1400   
    bRStrengthThreshold = 11  
#
# MAGIC - values for dt10 the where bedrock creates a pinch point for firn bands,is used by cleave().
#   The main problem is that the actual values are derived from parabola() which was developed later
#   and not yet fully integrated
#
peakPinchPoint = [2080,2140] 
pinchThreshold = 10
print(" peak from",peakPinchPoint[0],"to",peakPinchPoint[1],"currently hard coded")

#
# Analyse: Bedrock
#

retList = Analyse(sampleTrace,len(sampleTrace[0])-1,bedRockThreshold,bRStrengthThreshold,bRNumLines, ThinLines)
allLines0 = retList[0]
cleaveList=retList[1]
print("")
print(" Analyse: Bedrock lines  len allLines0",len(allLines0))
#for i in range(len(allLines0)):
#    print(i," ",len(allLines0[i]))
allLinesBed = []
for i in range(len(allLines0)):
    allLinesBed.append(allLines0[i])


if (OddityFlag == False):
    XStitchThreshold = 80
    YStitchThreshold = 550
else:
    XStitchThreshold = 40
    YStitchThreshold = 250    

if (ThinLines == True):
    allLinesBed = Mortar(allLinesBed, minLineLength, XStitchThreshold, YStitchThreshold, True)
    #for i in range(len(allLinesBed)):
    #    print(i," allLinesBed ",len(allLinesBed[i]), allLinesBed[i])
    bandLength =  30 # minimum length to graph
    newPinchPoint = parabola(allLinesBed,sampleTrace,bedRockThreshold,bandLength, 1)
    print(" new Pinch Point", newPinchPoint)
    print("")
    
if (OddityFlag == False):
    signalToNoiseRatio(dupRmvTrace,allLinesBed, 380, 2500, 1890, 2260, 10, 1500)
    
#
# create a Trace from the Bands
#
bandTrace = np.zeros([len(sampleTrace), len(sampleTrace[0])])
numLongLinesBed = makeGraphFromallLines(allLinesBed,bandTrace,bandLength,bedRockThreshold)

if (OddityFlag == False):
    makeStatsFromallLines(allLinesBed, bandLength, sampleTrace, phaseTrace, True, "Bedrock")
    
# More stats gathering
bedrockZoneDensity = bottomDensity(sampleTrace, bedRockThreshold)
print(" Density of bedrock zone:",bedrockZoneDensity)

# Graph Type and Maximum Colour
Type=('Paired',12)     # maxColour = 12
#Type=('tab20c',20)    # maxColour = 20
#Type=('gist_ncar',50) # maxColour = 50
#Type=('hsv',50)       # maxColour = 50
smoothLines = paintBand(sampleTrace,allLinesBed,bandLength)

print("  Bedrock lines",numLongLinesBed)
print("")
if (len(allLines0) > 0):
    smoothTrace = np.zeros([len(sampleTrace), len(sampleTrace[0])])
    makeGraphFromSmoothLines(smoothLines,smoothTrace,numLongLinesBed,Type[1])

    smoothTraceAvg = np.zeros([len(sampleTrace), len(sampleTrace[0])])
    makeGraphFromSmoothLinesAvg(smoothLines,smoothTraceAvg,numLongLinesBed,Type[1])

    graphData(bandTrace, "bedrock bandTrace", Type[0], 0 , False)
    graphData(smoothTraceAvg, "bedrock smoothTrace", Type[0], 0 , False)
else:
    print("bedrock graphs skipped as no lines found")
    
#
# Analyse: Surface bands
#

#
# Threshold value selections for Surface bands
#

if (OddityFlag == True):
    topLineTop = 600
    topLineBottom = 2000
    topLineThreshold = 1600
    topLineNumLines = 31
else:
    topLineTop = 200
    topLineBottom = 800
    topLineThreshold = 1200
    topLineNumLines = 101

retList = Analyse(sampleTrace, topLineBottom, topLineTop, topLineThreshold, topLineNumLines)
allLines1 = retList[0]
cleaveList=retList[1]
print("")
print(" Analyse: Surface bands  len allLines1",len(allLines1))
if (OddityFlag == False):
    makeStatsFromallLines(allLines1, minLineLength, sampleTrace, phaseTrace, True, "Top line")
#for i in range(len(allLines1)):
#    print(i," ",len(allLines1[i]))

#
# Analyse: Mid bands
#

#
# Threshold value selections for Mid bands
#
ThinLines=False

if (OddityFlag == True):
    midLineTop = 2100
    midLineBottom = 3300
    midLineThreshold = 48   # 40 -> 5 bands, 34 -> several dozen
    midLineNumLines = 561
else:
    #midLineTop = 312
    midLineTop = 280
    midLineBottom = 800
    midLineThreshold = 120
    midLineNumLines = 61

retList = Analyse(sampleTrace, midLineBottom, midLineTop, midLineThreshold, midLineNumLines)
allLines2 = retList[0]
cleaveList2 = retList[1] # unused
print("")
print(" Analyse: Mid lines  len allLines2",len(allLines2))
#for i in range(len(allLines2)):
#    print(i," ",len(allLines2[i]))


allLinesBand = []
for i in range(len(allLines1)):
    allLinesBand.append(allLines1[i])

for i in range(len(allLines2)):
    allLinesBand.append(allLines2[i])
    
print("")
print(" All Analysis: combined   len allLinesBand",len(allLinesBand))
#for i in range(len(allLinesBand)):
#    print(i," ",len(allLinesBand[i]))
    
if (OddityFlag == True):
    XStitchThreshold = 20
    YStitchThreshold = 50
else:
    XStitchThreshold = 80
    YStitchThreshold = 550
print("")
if (ThinLines == False):
    allLinesBand = Stitcher(sampleTrace, allLinesBand, minLineLength, maxNumLines, XStitchThreshold, YStitchThreshold)
    print("")
    allLinesBand = Wrangler(allLinesBand)
    print("")
    allLinesBand = Forfeiture(sampleTrace, allLinesBand, 3)
    print("")
#
# create a Trace from the Bands
#
bandTrace = np.zeros([len(sampleTrace), len(sampleTrace[0])])
if (OddityFlag == True):
    #signalToNoiseRatio(Trace, allLines,lineLen,yIdx,start,stop,interval,threshold=210,verbose=False)
    signalToNoiseRatio(dupRmvTrace,allLinesBand, int(len(Trace)*0.25), 2700,  0, 2000,25, 620,verbose=False)
#
# Translate lines to a new Graph
#

if (OddityFlag == True):
    bandLength = 1800 # minimum length to graph
else:
    bandLength = 500 # minimum length to graph
    ThinLines = True
print("")

numLongLinesBand = makeGraphFromallLines(allLinesBand,bandTrace,bandLength,bedRockThreshold+300)

if (OddityFlag == False):
    makeStatsFromallLines(allLinesBand, bandLength, sampleTrace, phaseTrace, False, "Mid line")

print("")
smoothLines = paintBand(sampleTrace,allLinesBand,bandLength)

if (ThinLines == False):
    retList = paintVoid(sampleTrace,smoothLines,allLinesBand,numLongLinesBand,bandLength)
    smoothLines = retList[0]
    allVoids = retList[1]
    for i in range(len(allVoids)):
        if (len(allVoids[i]) < 1):
            continue
    numLongLinesBand += retList[2]
    print("  allVoids numLongLinesBand",numLongLinesBand)
    if (OddityFlag == False):
        makeStatsFromallLines(allVoids, bandLength, sampleTrace, phaseTrace, False, "Voids")
else:
    if (OddityFlag == True):
        print(" Oddity: paintVoid skipped")
    else:
        print(" paintVoid not run due to ThinLine flag")

smoothTrace = np.zeros([len(sampleTrace), len(sampleTrace[0])])
makeGraphFromSmoothLines(smoothLines,smoothTrace,numLongLinesBand,Type[1])
print("")

smoothTraceAvg = np.zeros([len(sampleTrace), len(sampleTrace[0])])
makeGraphFromSmoothLinesAvg(smoothLines,smoothTraceAvg,numLongLinesBand,Type[1])
print("")

graphData(bandTrace, "bandTrace", Type[0], 0 , False)
graphData(smoothTraceAvg, "smoothTraceAvg", Type[0], 0 , False)
"""
retList = paintVoid(sampleTrace,smoothLines,allLinesBand,numLongLinesBand,bandLength)
smoothLinesVoid = retList[0]
allVoids = retList[1]
numLongLinesVoid = retList[2]
"""
allLinesBnB = []
for i in range(len(allLinesBand)):
    allLinesBnB.append(allLinesBand[i])
print(" allLinesBnB after Band",len(allLinesBnB))
"""
for i in range(len(allVoids)):
    if (len(allVoids[i]) < 1):
        continue
    allLinesBnB.append(allVoids[i])
print(" allLinesBnB after Voids",len(allLinesBnB))
"""
for i in range(len(allLinesBed)):
    allLinesBnB.append(allLinesBed[i])
print(" allLinesBnB after Bed",len(allLinesBnB))
if (OddityFlag == True):
    bandLengthBnB = 1800
else:
    bandLengthBnB = 30
    
bandTraceBnB = np.zeros([len(sampleTrace), len(sampleTrace[0])])
smoothTraceBnB = np.zeros([len(sampleTrace), len(sampleTrace[0])])

numLongLinesBnB = makeGraphFromallLines(allLinesBnB,bandTraceBnB,bandLengthBnB,bedRockThreshold)
smoothLinesBnB = paintBand(sampleTrace,allLinesBnB,bandLengthBnB)
#if (OddityFlag == True):
#    signalToNoiseRatio(dupRmvTrace,allLinesBnB,int(len(Trace)*0.25), 2800, 0,1800,25,660,verbose=False)
makeGraphFromSmoothLines(smoothLinesBnB,smoothTraceBnB,numLongLinesBnB,Type[1])
graphData(bandTraceBnB, "bandTraceBnB", Type[0], 0 , False)
graphData(smoothTraceBnB, "smoothTraceBnB", Type[0], 0 , False)

# Goldilock graph
retList = paintVoid(sampleTrace,smoothLinesBnB,allLinesBnB,numLongLinesBnB,bandLengthBnB)
smoothLinesBnB = retList[0]
allVoidsBnB = retList[1]
numLongLinesBnB += retList[2]
smoothTraceBnB = np.zeros([len(sampleTrace), len(sampleTrace[0])])
makeGraphFromSmoothLines(smoothLinesBnB,smoothTraceBnB,numLongLinesBnB,Type[1])
print("")
smoothTraceAvgBnB = np.zeros([len(sampleTrace), len(sampleTrace[0])])
makeGraphFromSmoothLinesAvg(smoothLinesBnB,smoothTraceAvgBnB,numLongLinesBnB,Type[1])
print("")

#graphData(bandTrace, "bandTrace", Type[0], 0 , False)
graphData(smoothTraceAvgBnB, "smootahTraceAvgBnB", Type[0], 0 , False)

endTime = datetime.datetime.now()
print("timex overall:", endTime-beginTime)
print("")


# In[ ]:





# In[ ]:





# In[ ]:




