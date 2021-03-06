import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import pyquaternion as qt #You will need to install this, sOrRy tRIsTaN <3
import sklearn.linear_model as lm
import seaborn as sns
from tqdm import tqdm as tqdm
from datetime import datetime
from math import cos, sin, asin, acos, atan2, tan
from math import radians as mrad
from numba import njit, jit
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import *
from celluloid import Camera




fileName = "Data_new_encoding.csv"
unconfirmedFileName = "Unconfirmed_exoplanet_data.csv"

# ------------------------------ Definition of some constants --------------------------#
moonOrbitalPeriod = 28 # Days
hrsPerExo = 150 # Hours of observation time
raOfAscNode = 125.08 # Right ascension of the ascending node - Constant value for lunar orbit
initialArgOfPeriapseMoon = 60.78357656807530 # Argument of Periapse of the Moon - Constant value based on J2000.
craterLatitude = -34 # Latitude of the location on the Moon
craterLongitude = 152.0 # Longitude of the location on the Moon


# ------------------------------ Definition of quantile plot parameters ----------------#

quantileExo = round(hrsPerExo/(moonOrbitalPeriod*24),3)
print(f"Exoplanet quantile = {quantileExo}")
quantilePlusTwentyFive = round(1.25*quantileExo,3)
quantileMinusTwentyFive = round(0.75*quantileExo,3)

#Resolution of bars on the plot (in degrees)
barReso = 5
# The quantiles you want
quantiles = [quantileMinusTwentyFive,quantileExo,quantilePlusTwentyFive]
colours = ["blue", "orange", "green", "red", 'purple'] # Colours used for the quantile bars
unconfirmed_colours = ["yellow", "lime", "pink"] # Colours used for unconfirmed exoplanets



# ------------------------------ Defining simulation parameters ------------------------#


#Days since January 1st, 2000. (J2000)
timeInDays = 9131 # January 1st, 2025
timeInCenturies = timeInDays/(365.2425*100)

#Mission length
lengthInDays = 6794 # 18.6 years (Full axial precession) # 6794




# ------------------------------------------------- Lunar Orbit Simulation ------------------------------------------------------#


# ---- Calculate initial position

moonOrbitalPeriodDays = (2.328185776517964*10**6)/(24*60**2)  # Based on JPL Ephemeris for January 1st, 2000
daysIntoPeriod = timeInDays%moonOrbitalPeriodDays # Returns remainder of the operation
angularRateMoon = 360 / moonOrbitalPeriodDays
argumentOfPeriapseMoon = daysIntoPeriod*angularRateMoon + initialArgOfPeriapseMoon
moonRightAscension = raOfAscNode + argumentOfPeriapseMoon

if moonRightAscension >= 360:
    moonRightAscension -= 360


def lunarOrbitRASimulation(duration:float):
    """Simulates the right ascension of a perfectly circular lunar orbit. The circular orbit is an assumption
    that might be modified in the future.

    Args:
        duration (float): Duration of the mission (lengthInDays)

    Returns:
        data (np.array): Array with column 0 containing the number of days from the beginning of
        the simulation, and column 1 containing the right ascension of the Moon. 
    """
    #data1 = np.transpose(np.linspace(0,duration,(duration+1)))
    data1 = np.linspace(0,duration,(duration))
    data2 = np.linspace(0, duration,(duration))
    data2 = data2%moonOrbitalPeriodDays
    data2 = (data2*angularRateMoon) + moonRightAscension
    data2[np.where(data2 >= 360)] = data2[np.where(data2 >= 360)] - 360
    data = np.vstack([data1,data2]).T


    return data


def lunarNorthPolePosition(epochs:np.array):
    """Calculates the position of the Moon's north pole according to the International Astronomical Union's
    model. This is used to later calculate the effects of axial precession on the overall FOV.

    For details on the model, visit:

    https://astropedia.astrogeology.usgs.gov/alfresco/d/d/workspace/SpacesStore/28fd9e81-1964-44d6-a58b-fbbf61e64e15/WGCCRE2009reprint.pdf

    Args:
        epochs (np.array): The epoch according to the J2000 reference(the length in days added onto the epoch at the beginning of the mission)

    Returns:
        data (np.array): Array containing the right ascension and declination of the lunar north pole.
    """
    E1 = (epochs*-0.0529921) + 125.045
    E2 = (epochs*-0.1059842) + 250.089
    E3 = 13.0120009*epochs + 260.008
    E4 = 13.3407154*epochs + 176.625
    E5 = 0.9856003*epochs + 357.529
    E6 = 311.589 + 26.4057084*epochs
    E7 = 134.963 + 13.0649930*epochs
    E8 = 276.617 + 0.3287146*epochs
    E9 = 34.226 + 1.7484877*epochs
    E10 = 15.134 - 0.1589763*epochs
    E11 = 119.743 + 0.0036096*epochs
    E12 = 239.961 + 0.1643573*epochs
    E13 = 25.053 + 12.9590088*epochs

    right_ascension = 269.9949 + (0.0031/(365.2425*100))*epochs - 3.8787*np.sin(np.radians(E1)) \
         - 0.1204*np.sin(np.radians(E2)) +0.0700*np.sin(np.radians(E3)) - 0.0172*np.sin(np.radians(E4)) \
             +0.0072*np.sin(np.radians(E6)) - 0.0052*np.sin(np.radians(E10)) + 0.0043*np.sin(np.radians(E13))

    declination = 66.5392 + (0.0130/(365.2425*100))*epochs + 1.5419*np.cos(np.radians(E1)) \
        + 0.0239*np.cos(np.radians(E2)) - 0.0278*np.cos(np.radians(E3)) + 0.0068*np.cos(np.radians(E4)) \
            - 0.0029*np.cos(np.radians(E6)) + 0.0009*np.cos(np.radians(E7)) + 0.0008*np.cos(np.radians(E10))\
                - 0.0009*np.cos(np.radians(E13))

    data = np.vstack([right_ascension,declination]).T

    return data


def incorporateAxialPrecession(northPoleAngles:np.ndarray) -> np.array:
    """ Transforms the unit vector [1,0,0] to the angles of the north pole given by the IAU model. Yields a series of
    vectors marking the position of the north pole.

    Args:
        northPoleAngles (np.ndarray): Angles given by the IAU model, output of lunarNorthPolePosition

    Returns:
        vectors (np.array): Vectors pointing towards the north pole over the given time of the mission.
    """
    vectors = np.zeros([len(northPoleAngles[:,0]),3])

    unit_vector = np.array([1,0,0])

    for i in range((len(northPoleAngles[:,0]))):

        q1 = qt.Quaternion(axis=[0,0,1],angle = np.radians(northPoleAngles[i,0]))
        q2 = qt.Quaternion(axis=[0,-1,0],angle = np.radians(northPoleAngles[i,1]))
        q3 = q1*q2
        vector = q3.rotate(unit_vector)
        vectors[i] = vector

    return vectors


def generateTelescopeVectors(polarPositionVectors:np.ndarray, rightAsc) -> np.array:
    """Transforms the vectors provided by incorporateAxialPrecession and rotates them by the
    latitude, and longitude of the crater and right ascension of the Moon. This represents the vectors perpendicular
    to the surface of the crater at a given point in time.

    Args:
        polarPositionVectors (np.ndarray): Array containing vectors representing the position of the north pole.
        rightAsc ([type]): Array containing the right ascension of the Moon in a given point of its orbit.

    Returns:
        vectors (np.array): Vectors perpendicular to the surface of the crater at different points in time.
    """

    vectors = np.zeros([len(polarPositionVectors[:,0]),3])


    for i in range(len(polarPositionVectors[:,0])):

        q1 = qt.Quaternion(axis=polarPositionVectors[i,:],angle = np.radians(craterLongitude) + np.radians(rightAsc[i,1]))
        new_vector_x = 1*np.cos(np.radians(rightAsc[i,1]+craterLongitude))
        new_vector_y = 1*np.sin(np.radians(rightAsc[i,1]+craterLongitude))
        new_vector_z = (-new_vector_x*polarPositionVectors[i,0] - new_vector_y*polarPositionVectors[i,1])/polarPositionVectors[i,2]
        orth = np.array([new_vector_x,new_vector_y,new_vector_z])

        q2 = qt.Quaternion(axis=orth, angle = np.radians(craterLatitude) - np.radians(90))
        q3 = q2*q1

        vector = q3.rotate(polarPositionVectors[i,:])
        vectors[i,:] = vector

    for i in range(len(polarPositionVectors[:,0])):

        q1 = qt.Quaternion(axis = polarPositionVectors[i,:], angle = np.radians(angularRateMoon*i))

        vector = q1.rotate(vectors[i,:])
        vectors[i,:] = vector

    return vectors


def find_angles(rotatedPosition:np.ndarray) -> np.array:
    """Finds the angles between the given vectors and the coordinate system, in order to determine the right ascension and declination
    for plotting purposes.

    Note: For now, this only works for the southern hemisphere. This can be corrected by changing the sign towards the end of the function
    (It is indicated). Will include a switch in the future.

    Args:
        rotatedPosition (np.ndarray): Crater position vectors, adjusted to axial precession.

    Returns:
        angles (np.array): Gives the corrected right ascension and declination of the crater, displaying the path visible to the telescope.
    """

    xyPlaneNormal = np.array([0,0,1]).T
    xzModulus = np.zeros([len(rotatedPosition[:,0])])
    xz_projection = np.delete(rotatedPosition, 2, 1)
    xUnitVector = np.zeros([len(rotatedPosition[:,0]),2])

    for i in range(len(rotatedPosition[:,0])):
    
        xzModulus[i] = np.linalg.norm(xz_projection[i,:])
        xUnitVector[i,:] = [1,0]

    angleRA = np.zeros([len(rotatedPosition[:,0]),1])

    for i in range(len(rotatedPosition[:,0])):


        angle = atan2(xz_projection[i,1],xz_projection[i,0])
        angleRA[i] = np.degrees(angle)
        #dot = np.dot(xz_projection[i,:], xUnitVector[i,:].T)
        #ar = np.vstack([xz_projection[i,:],xUnitVector[i,:].T])
        #det = np.linalg.det(ar)
        #angleRA[i] = np.degrees(atan2(det,dot))
        #angleRA[i] = np.degrees(np.arccos(np.dot(xz_projection[i,:],xUnitVector[i,:].T) / xzModulus[i]))

    
    
    angleDec = np.degrees(np.arcsin(np.dot(rotatedPosition,xyPlaneNormal))) # Formula from my high school maths textbook
    

    angles = np.vstack([angleRA.T,angleDec]).T
    angles[:,0][np.where(angles[:,1] > 90)] = angles[:,0][np.where(angles[:,1] > 90)] + 180
    angles[:,1][np.where(angles[:,1] > 90)] = 90 - (angles[:,1][np.where(angles[:,1] > 90)] - 90)
    angles[:,0][np.where(angles[:,0] < 0)] = 360 + angles[:,0][np.where(angles[:,0] < 0)]
    angles[:,0][np.where(angles[:,0] > 360)] = angles[:,0][np.where(angles[:,0] > 360)] - 360
    """print(max(angleRA))
    # Hotfix
    ind = np.argmax(angles[0,:], axis=0)
    print(angles[0,ind])
    angles[0,ind:] = 360 - angles[0,ind:]"""

    # This only fixes the problem for one orbit; you need to make it work for all of them.


    '''x_axis = [1,0,0]
    z_axis = [0,0,1]

    xy_projection = np.delete(rotatedPosition, 2, 1)
    
    xUnitVector = np.zeros([len(rotatedPosition[:,0]),2])
    
    

    angles = np.zeros([len(rotatedPosition[:,0]),2])
    xyModulus = np.zeros([len(rotatedPosition[:,0])])
    

    for i in range(len(rotatedPosition[:,0])):
    
        xyModulus[i] = np.linalg.norm(xy_projection[i,:])
        xzModulus[i] = np.linalg.norm(xz_projection[i,:])
        xUnitVector[i,:] = [1,0]
        zUnitVector[i,:] = [0,1]

        # FIND ANGLE BETWEEN VECTORS

    print(xUnitVector)
    print(zUnitVector)

    for i in range(len(rotatedPosition[:,0])):

        angles[i,0] = np.degrees(np.arccos(np.dot(xy_projection[i,:], xUnitVector[i,:].T) / xyModulus[i]))
        angles[i,1] = np.degrees(np.arccos(np.dot(xz_projection[i,:],xUnitVector[i,:].T) / xzModulus[i]))

    angles[:,0][np.where(angles[:,1] > 90)] = angles[:,0][np.where(angles[:,1] > 90)] + 180
    angles[:,1][np.where(angles[:,1] > 90)] = 90 - (angles[:,1][np.where(angles[:,1] > 90)] - 90)
    #decMoon[np.where(decMoon < -90)] = -decMoon[np.where(decMoon < -90)] - 180

    angles[:,1] = -angles[:,1] # Change sign here for south-north hemisphere (South: -, North: +)

    print(angles)
    print(f'Min RA: {min(angles[:,0])}')
    print(f'Min DEC: {min(angles[:,1])}')
    print(f'Max RA: {max(angles[:,0])}')
    print(f'Max DEC: {max(angles[:,1])}')'''

    return angles.T


def getGamma(tel:np.ndarray, exos:np.ndarray) -> np.ndarray:
    """Get gamma for all combinations

    Args:
        a (np.ndarray): rTelescope
        b (np.ndarray): rExos

    Returns:
        np.ndarray: gamma for all combinations
    """
    return np.degrees(np.arccos(np.dot(tel,exos)))


def anglesToVector(ra:np.ndarray, dec:np.ndarray) -> np.ndarray:
    """Convert angles to unit vectors

    Args:
        ra (np.ndarray): right ascension
        dec (np.ndarray): declination

    Returns:
        np.ndarray: vector
    """

    unit_vector = np.array([1,0,0])
    ra = np.radians(ra)
    dec = np.radians(dec)

    vectors = np.zeros([len(ra),3])

    for i in range(len(ra)):

        q1 = qt.Quaternion(axis=[0,0,1],angle = ra[i])
        q2 = qt.Quaternion(axis=[0,-1,0], angle = dec[i])
        q3 = q1*q2
        vector = q3.rotate(unit_vector)
        vectors[i] = vector

    return vectors


def quantileData(qquantile, colorBar, DF:pd.DataFrame, MinDF:pd.DataFrame,MaxDF:pd.DataFrame,DiffDF:pd.DataFrame):
    """quantileData alculates the number of visible exoplanets as a function of pointing angle for different quantiles. 

    Args:
        qquantile ([type]): An array of quantiles to calculate
        colorBar ([type]): Colours given to the quantiles
        DF (pd.DataFrame): Gamma angle dataframe
        MinDF (pd.DataFrame): Minimum gamma per planet dataframe
        MaxDF (pd.DataFrame): Maximum gamma per planet dataframe
        DiffDF (pd.DataFrame): Difference between min and max datagrame
    """    
    exoQuantile = pd.DataFrame(DF.quantile(qquantile, axis=0).rename("Quantile")).transpose()
    exoQuantile = exoQuantile.append(MinDF.iloc[0].rename("Min"), ignore_index=False)
    exoQuantile = exoQuantile.append(MaxDF.iloc[0].rename("Max"), ignore_index=False)
    exoQuantile = exoQuantile.append(DiffDF.iloc[0].rename("Diff"), ignore_index=False)

    for angle in [x*barReso for x in range(0,(int(90/barReso) + 1))]:
        exoQuantile = exoQuantile.append(exoQuantile.loc["Quantile"].rename(f"{angle}") <= angle, ignore_index=False)
        


    sumVisible = ((exoQuantile == 1).astype(int).sum(axis=1)).iloc[4::]

    exoQuantile.to_csv(f"data/{qquantile}.csv")
    sumVisible.to_csv(f"data/{qquantile} Numbers.csv")
    plt.bar(sumVisible.index, sumVisible.values, color=colorBar, label=f"q = {quantile} cycle")


# ------------------------------------------------ RUN PROGRAM -------------------------------------- #


# Execute lunar orbit simulation --------
RA = lunarOrbitRASimulation(lengthInDays) 

# Define epoch ---------
epoch = np.linspace(0,lengthInDays, lengthInDays) + timeInDays

# Determine north pole movement from IAU model ---------

plt.rc('font', size=32)
plt.rc('axes', titlesize=32)     # fontsize of the axes title
plt.rc('axes', labelsize=32)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=32)    # fontsize of the tick labels
plt.rc('ytick', labelsize=32)    # fontsize of the tick labels
plt.rc('legend', fontsize=26)    # legend fontsize

moonNorthPoleAngles = lunarNorthPolePosition(epoch)
averageRA = np.average(moonNorthPoleAngles[:,0])
print(f"Average Polar Right Ascension: {averageRA}")
fig, ax = plt.subplots()
ax.plot(moonNorthPoleAngles[:,0],moonNorthPoleAngles[:,1], label='Lunar north pole orientation')
plt.title('Lunar Axial Precession according to the IAU model')
ax.grid(which='both')
plt.ylabel('Declination [??]')
plt.xlabel('Right Ascension [??]')
ax.legend()
ax.set_aspect('equal')
plt.show()

# Create vectors for north pole position -------
rotatedPolarVectors = incorporateAxialPrecession(moonNorthPoleAngles)

# Rotate from north pole to telescope for telescope position -------
telescopeVectors = generateTelescopeVectors(rotatedPolarVectors,RA)


# ------------- 3D plotting of the path -----------------------#
# Plot spherical plot -------
u, v = np.mgrid[0:2*np.pi:25j, 0:np.pi:25j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)



# Plot path on sphere -------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# make the panes transparent
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# make the grid lines transparent
ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.plot3D(telescopeVectors[:,0],telescopeVectors[:,1],telescopeVectors[:,2], color='red', linewidth = 0.2)
ax.plot_wireframe(x, y, z, color="grey", alpha = 0.4)
plt.show()


# ----------- Plot 2D RA-Dec plot -------------

# Data reading ---------
data = pd.read_csv(fileName, sep=',', skiprows=38)
exo_data = data[['ra', 'dec', 'pl_name']]

unconfirmed_data = pd.read_csv(unconfirmedFileName, sep=',', skiprows=50)
unconfirmed_exo_data = unconfirmed_data[['ra','dec','pl_name']]

# Find RA and Dec -----------
visionAngles = find_angles(telescopeVectors)

# Plot --------

plt.rc('font', size=32)          # controls default text sizes
#plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
#plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
#plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
#plt.rc('figure', titlesize=BIGGER_SIZE)

ax = plt.subplot()
plt.scatter(visionAngles[0,:],visionAngles[1,:], label='Telescope vector path')
plt.title('Debug plot')
plt.legend()
plt.grid(which='both')
plt.xlabel('Right Ascension [??]')
plt.ylabel('Declination [??]')
ax.set_xlim(0,360)
plt.show()

np.savetxt("visionAngles.csv", visionAngles, delimiter=",")


fig, ax = plt.subplots()

sns.set_theme()
ax.set_ylim(-90,90)
ax.set_xlim(0,360)
ax.set_facecolor("k")
hlines = np.linspace(-90,90,5)
vlines = np.linspace(0,360,6)
# grid
plt.hlines(hlines, 0,360, color= "gray", linestyles="--",zorder=0, linewidths=0.5)
plt.vlines(vlines, -90, 90, color= "gray", linestyles="--",zorder=0, linewidths=0.5)

#tele
plt.scatter(visionAngles[0,:], visionAngles[1,:], c = "r", s = 0.1, linewidth=3)
# exo
plt.scatter(data['ra'], data['dec'], s=1.5, marker="*",
            c="white", alpha=0.6, label=None)
#galaxy
galactic = np.zeros(1000)
galactic_angle = np.linspace(0,360,1000)
for i in range(1000):

    galactic[i] = 60*cos(galactic_angle[i]*np.pi/180)
plt.plot(galactic_angle, galactic, linestyle='--', label="Galactic plane", linewidth=3)
# dummy
plt.plot((-100, -90), (-180, -170),c = "r", label="Telescope path", linewidth=3)
plt.scatter(-100, -180, marker="*",
            c="white",label="Confirmed exoplanets")
plt.ylabel("Declination [??]")
plt.xlabel("Right Ascension [??]")
plt.title("Observable region")
plt.legend()
plt.show()


# ----------------------- Exoplanet visibility --------------------------



print(f"Total exoplanets = {len(exo_data)}")
print(f"Total exoplanets below 18 dec = {len(exo_data[exo_data['dec'] < 18])}")

print(f'Total unconfirmed exoplanets = {len(unconfirmed_exo_data)}')
print(f'total unconfirmed exoplanets below 18 dec = {len(unconfirmed_exo_data[unconfirmed_exo_data["dec"] < 18])}')

# exo_data = exo_data[exo_data.dec <= 3]
exo_ra = exo_data.iloc[:, 0].to_numpy()
exo_dec = exo_data.iloc[:, 1].to_numpy()
exo_names = exo_data.iloc[:, 2].to_list()

unconfirmed_exo_ra = unconfirmed_exo_data.iloc[:,0].to_numpy()
unconfirmed_exo_dec = unconfirmed_exo_data.iloc[:,1].to_numpy()
unconfirmed_exo_names = unconfirmed_exo_data.iloc[:,2].to_list()


# Turn exoplanet angular positions into position vectors -----------
rExos = anglesToVector(exo_ra, exo_dec).T
rUnconfirmedExos = anglesToVector(unconfirmed_exo_ra,unconfirmed_exo_dec).T


# Find angles between telescope and every single exoplanet -----------
gamma = getGamma(telescopeVectors, rExos)
gammaMin = pd.DataFrame([np.min(gamma, axis=0)], columns=exo_names)
gammaMax = pd.DataFrame([np.max(gamma, axis=0)], columns=exo_names)
gammaDiff = pd.DataFrame([np.max(gamma, axis=0)- np.min(gamma, axis=0)], columns=exo_names)


gammaDF = pd.DataFrame(gamma, columns=exo_names)


#Unconfirmed exoplanets

unconfirmedGamma = getGamma(telescopeVectors,rUnconfirmedExos)
unconfirmedGammaMin = pd.DataFrame([np.min(unconfirmedGamma, axis=0)], columns=unconfirmed_exo_names)
unconfirmedGammaMax = pd.DataFrame([np.max(unconfirmedGamma, axis=0)], columns=unconfirmed_exo_names)
unconfirmedGammaDiff = pd.DataFrame([np.max(unconfirmedGamma, axis=0)- np.min(unconfirmedGamma, axis=0)], columns=unconfirmed_exo_names)

unconfirmedGammaDF = pd.DataFrame(unconfirmedGamma, columns=unconfirmed_exo_names)


# Compare every exoplanet angle with different FOV ------------------


for quantile, color in zip(quantiles, colours):
    quantileData(quantile, color, gammaDF, gammaMin, gammaMax, gammaDiff)

'''Uncomment for confirmed and unconfirmed exoplanets. WARNING: I do not take responsibility for any damage to your retinas caused by the colour scheme. ~N.'''

#for quantile, colour in zip(quantiles, unconfirmed_colours):
    #thingy(quantile, colour, unconfirmedGammaDF, unconfirmedGammaMin, unconfirmedGammaMax, unconfirmedGammaDiff)



plt.legend()
plt.title('Confirmed Exoplanets in sight as a function of pointing angle')
plt.xlabel('Pointing angle [??]')
plt.ylabel('Number of visible exoplanets')
plt.grid(which='both', axis='y')
plt.show()


# ------------------------------------------------- Animation ------------------------------------------- #


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)
ax.plot_wireframe(x, y, z, color="grey", alpha = 0.2)
#ax.plot_surface(x, y, z, color="whitesmoke", alpha = 0.4)

pathX = []
pathY = []
pathZ = []

lineOrigin = []
lineX = [0]
lineY = [0]
lineZ = [0]

lineX.append(telescopeVectors[0,0])
lineY.append(telescopeVectors[0,1])
lineZ.append(telescopeVectors[0,2])

pathX.append(telescopeVectors[0,0])
pathY.append(telescopeVectors[0,1])
pathZ.append(telescopeVectors[0,2])

#Polar vector

poleX = []
poleY = []
poleZ = []

poleX.append(-rotatedPolarVectors[0,0])
poleX.append(rotatedPolarVectors[0,0])
poleY.append(-rotatedPolarVectors[0,1])
poleY.append(rotatedPolarVectors[0,1])
poleZ.append(-rotatedPolarVectors[0,2])
poleZ.append(rotatedPolarVectors[0,2])

# Create FOV limit:

fovAngle = 85
rFOV  = np.zeros([len(telescopeVectors[:,0]),3])

unit_vector = np.array([1,0,0])

FOVvectors = np.zeros([len(telescopeVectors[:,0]),3])


for i in range(len(rotatedPolarVectors[:,0])):

    q1 = qt.Quaternion(axis=[0,0,1],angle = np.radians(craterLongitude) + np.radians(RA[i,1]))
    q2 = qt.Quaternion(axis=[0,-1,0], angle = np.radians(craterLatitude) + np.radians(fovAngle) - np.radians(90))
    q3 = q1*q2
    vector = q3.rotate(rotatedPolarVectors[i,:])
    FOVvectors[i,:] = vector





line, = ax.plot3D(lineX,lineY,lineZ, color="red", label='Observation path')
path, = ax.plot3D(pathX,pathY,pathZ, color="red")
fov, = ax.plot3D(FOVvectors[0,0],FOVvectors[0,1],FOVvectors[0,2], color='blue', label=f'FOV Limit: {fovAngle} degrees')
polepath, = ax.plot3D(rotatedPolarVectors[0,0],rotatedPolarVectors[0,1],rotatedPolarVectors[0,2], color = 'black', label='Pole Precession')
pole, = ax.plot3D(poleX,poleY,poleZ, color = 'black')





'''It's not elegant, but it works'''

def animate(i): 

    global vectors
    vectors = telescopeVectors

    lineX[1] = vectors[i,0]
    lineY[1] = vectors[i,1]
    lineZ[1] = vectors[i,2]
    lineXArray = np.array(lineX)
    lineYArray = np.array(lineY)
    lineZArray = np.array(lineZ)

    pathX.append(vectors[i,0])
    pathY.append(vectors[i,1])
    pathZ.append(vectors[i,2])
    pathXArray = np.array(pathX)
    pathYArray = np.array(pathY)
    pathZArray = np.array(pathZ)


    poleX[0] = -rotatedPolarVectors[i,0]
    poleX[1] = rotatedPolarVectors[i,0]
    poleY[0] = -rotatedPolarVectors[i,1]
    poleY[1] = rotatedPolarVectors[i,1]
    poleZ[0] = -rotatedPolarVectors[i,2]
    poleZ[1] = rotatedPolarVectors[i,2]

    poleXArray = np.array(poleX)
    poleYArray = np.array(poleY)
    poleZArray = np.array(poleZ)

    line.set_xdata(lineXArray)
    line.set_ydata(lineYArray)
    line.set_3d_properties(lineZArray)

    path.set_xdata(pathXArray)
    path.set_ydata(pathYArray)
    path.set_3d_properties(pathZArray)

    fov.set_xdata(FOVvectors[0:i,0])
    fov.set_ydata(FOVvectors[0:i,1])
    fov.set_3d_properties(FOVvectors[0:i,2])

    polepath.set_xdata(rotatedPolarVectors[0:i,0])
    polepath.set_ydata(rotatedPolarVectors[0:i,1])
    polepath.set_3d_properties(rotatedPolarVectors[0:i,2])

    pole.set_xdata(poleXArray)
    pole.set_ydata(poleYArray)
    pole.set_3d_properties(poleZArray)

    




animation = FuncAnimation(fig, func=animate, frames = np.arange(0,lengthInDays,1), interval = 0.01, save_count=6794, repeat=False)


plt.title('Observation angle motion (Period: 18.6 years)')
plt.legend()
plt.show()





f = r"animation.gif" 
writergif = PillowWriter(fps=24) 
animation.save(f, writer=writergif)



""" fig = plt.figure(figsize=(8, 8))
m = Basemap(projection='lcc', resolution=None,
            lon_0=0, lat_0=50, lat_1=45, lat_2=55,
            width=1.6E7, height=1.2E7)
draw_map(m) """





""" plt.show() """



""" def lineAnimationFrame(i):

    linePlot[1,:] = telescopeVectors[i,:]

    line.set_xdata(linePlot[:,0])
    line.set_ydata(linePlot[:,1])
    line.set_3d_properties(linePlot[:,2])

    
    return line,

def pathAnimationFrame(i):

    global pathPlotGlobal
    if i == 1:
        pathPlotGlobal = pathPlot
        
    pathPlotTemp = np.vstack([pathPlotGlobal, telescopeVectors[i]])
    pathPlotGlobal = pathPlotTemp
    path.set_xdata(pathPlotTemp[i,0])
    path.set_ydata(pathPlotTemp[i,1])
    path.set_3d_properties(pathPlotTemp[i,2])

    return path,

pathFrame = np.arange(1,lengthInDays,1)"""




