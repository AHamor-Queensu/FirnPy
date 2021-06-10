# FirnPy
Queen's University undergraduate physics thesis project for Phys 590 by Alexander Hamor under the supervision of Dr. Laura Thomson.

Documentation of all the functions developed for FirnPy.

parseHD(text_file)
""" Parse the .HD ASCII header file.

Parameters:
      text_file: String containing the path, filename and extension of the .HD file

Function:
    Parse the ASCII strings in the HD file for pertinent strings which are stored in a ‘hd’ dictionary

Return Values:
    hd - dictionary of needed or interesting values
    twtt - an np array of two way travel times for the Trace
"""

parseDTn(binary_file)
""" Parse the .DT1 binary data file.

Parameters:
binary_file: String containing the path, filename and extension of the .DT1 file

Function:
    Parse the ASCII and the binary fields in the DT1 file for the strength array. Other values are parsed but do not appear to be needed.

Return Values:
    SA - strength array of integer values in a list of lists
"""

window(Trace, upperX, upperY, lowerX, lowerY, Data=False)
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

graphData(Trace, Title, cmap_type = 'seismic', Gap=200, someData=True)
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

removeAverage(Trace, threshold, numTraces=1)
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

detectDuplicates(winTrace, threshold=0)
""" Check for 80% similarities within a thin threshold.

Parameters:
winTrace: a sample strength array returned from window()
threshold: the maximum allowable variation (positive)

Function:
    If, perchance, a snowmobile dragging a GPR unit stops to retrieve a notebook and the unit is left running creating line artifacts, list them.

Return Values:
    removal_list - the list Trace columns that have been detected ass duplicates - see removeDuplicates for their fate

removeDuplicates(Trace, removal_list)
""" Remove a list of duplicates from the Trace.

Parameters:
Trace: strength array as a list of list
removal_list: a list of lines that are duplicates 

Function:
    Delete duplicate columns from Trace

Return Values:
    window - the section of the array
"""

verticalBandPass(hd, data, low, high, order=5, filttype='butter', cheb_rp=5,fir_window='hamming')
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
Cheb_rp: the Chebyshev type filter level
Fir_window: the finite impulse response filter (fir) window type

Function:
    Wanting a bandpass filter and being way over my head, the kind folks at ImpDar and GPRpy had sources on line to guide these efforts in using scipy functionality. Many thanks. 

Return Values:
    window - the section of the array
"""

lookForBand(Trace, Bands, yp=800, ystop=0, depth=5, width=7, threshold=1200)
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
    
"""

lookForLines(Bands, nBands = 61, maxNumLines = 21, minLineLength = 40)
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

findLine(Bands, nBands, line, xIdx, pointsFound, pointsGapped)
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

cleave(Bands, Bands_shape, threshold, cleaveList, cleaveThreshold = 25, peakPinchPoint = [2080,2140], peakPinchThreshold = 10)
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

makeGraphFromallLines(allLines, bandTrace, minLength, threshold=0, maxColour=12)
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

makeGraphFromSmoothLines(smoothLines, smoothTrace, noLines, maxColour)
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

makeGraphFromSmoothLinesAvg(smoothLines, smoothAvgTrace, noLines, maxColour)
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


checkForSpikes(smoothLines, x, line, cleaved)
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


paintBand(Trace, allLines, minLineLength, numBands=1000)
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


paintVoid(Trace, smoothLines, allLines, numLongLines, minLineLength)
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


Stitcher(Trace, allLines, minLineLength, maxNumLines, XStitchThreshold=25, YStitchThreshold=60)
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


Wrangler(allLines)
""" Tries to restore the order of the lines. 

Parameters:
allLines: a list of list of lines

Function:
Wrangler applies the Time Team axiom that the top layers are younger than the lower layers, excepting folding, which is quite possible, bands will be deposited over time so when the lines cross it is most likely a touch but this confuses the algorithm, Wrangler restores the lines being deposited in order.

Return Values:
    allLines, a list of list of lines
"""


Forfeiture(Trace, allLines, lawLine=0)
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


Mortar(allLines, minLineLength, XMortThreshold=40, YMortThreshold=150, bedrock=True)
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


Analyse(Trace, ystart=800, ystop=200, threshold=1200, nBands=61, thin=False)
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


makeStatsFromallLines(allLines, minLength, Trace, phaseTrace, differential=False, Tag="")
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


cycleFill(k, startIdx, stop, center, maxCycleLen, phases, phaseData, invPhaseData, phaseDecadeData, cycleHist, cycleRegHist, decadeHist, decadeRegHist, Tag)
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


unWave(Trace, maxValue)
""" Gain function which corrects for the wave nature of the signal. 

Parameters:
Trace: list of lists of backscatter/strengths
maxValue: used for graphs called within function (fix)

Function:
Build a list of arcsine values based on cycle length, and multiply the strengths by the arcsines to correct them.

Return Values:
    phaseData,invPhaseData
"""


parabola(allLines, Trace, threshold, bandLength=50, style=0)
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


bottomDensity(Trace, threshold)
""" Calculate the bedrock backscatter strength. 

Parameters:
Trace: list of list of strengths/backscatter
threshold: the density of the Band for the bottom

Function:
    Calculate the density of the bottom parabola.

Return Values:
    density
"""


harVar(Trace, threshold=100, zeroPoint=1)
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


varAverage(Trace, threshold=100, zeroPoint=1)
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


varAverage2(Trace, maxValue=10, zeroPoint=1)
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


signalToNoiseRatio(Trace,allLines,lineLen,yIdx,start,stop,interval,threshold=210)
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
