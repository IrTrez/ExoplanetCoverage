import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import pyquaternion as qt #You will need to install this, sOrRy tRIsTaN <3
import sklearn.linear_model as lm
from tqdm import tqdm as tqdm
from datetime import datetime
from math import cos, sin, asin, acos, atan2, tan
from math import radians as mrad
from numba import njit, jit



fileName = "Data_new_encoding.csv"
unconfirmedFileName = "Unconfirmed_exoplanet_data.csv"

moonOrbitalPeriod = 28 # Days
hrsPerExo = 150 # Hours of observation time
quantileExo = round(hrsPerExo/(moonOrbitalPeriod*24),3)
print(f"Exoplanet quantile = {quantileExo}")
quantilePlusTen = round(1.25*quantileExo,3)
quantileMinusTen = round(0.75*quantileExo,3)

#Resolution of bars on the plot (in degrees)
barReso = 5
# The quantiles you want
quantiles = [quantileMinusTen,quantileExo,quantilePlusTen]
#Resolution of the right ascension path traced. 360 = 1 computation per degree.
raResolution = 360
colours = ["blue", "orange", "green", "red", 'purple'] # Colours used for the quantile bars
unconfirmed_colours = ["yellow", "lime", "pink"] # Colours used for unconfirmed exoplanets

ra_of_asc_node = 125.08 # Constant value for lunar orbit
craterLatitude = -88.5 # Latitude of the location on the Moon
craterLongitude = 152.0 # Longitude of the location on the Moon
A = 16.56 # 'Amplitude' of the movement induced by libration (inclination of the Moon, assumed constant and assumed orientation)




def teleview(raBasic: np.ndarray, lat: float = craterLatitude, lon: float = craterLongitude, A: float = A, an: float = ra_of_asc_node) -> np.ndarray:
    """Get declination of given position in array form.

    Args:
        craterLatitude (float): Crater latitude
        A (float): Amplitude Moon tilt
        ra_basic (np.ndarray): Right Ascension
        an (float): Right Ascension of ascending node

    Returns:
        np.ndarray: transformed array
    """
    return lat - (A * np.sin(np.radians(raBasic) + mrad(an) + mrad(lon-180)))


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
        q2 = qt.Quaternion(axis=[0,1,0], angle = dec[i])
        q3 = q1*q2
        vector = q3.rotate(unit_vector)
        vectors[i] = vector

    return vectors

    #return np.array([np.cos(ra) * np.cos(dec), np.sin(ra), -np.cos(ra)*np.sin(dec)]).T


def getGamma(tel:np.ndarray, exos:np.ndarray) -> np.ndarray:
    """Get gamma for all combinations

    Args:
        a (np.ndarray): rTelescope
        b (np.ndarray): rExos

    Returns:
        np.ndarray: gamma for all combinations
    """
    return np.degrees(np.arccos(np.dot(tel,exos)))

def regressionHammer(exoData:pd.DataFrame):
    """

    'When you have a hammer, everything looks like a nail'

    Models the number of exoplanets visible as a function of the angle. Could be applied to estimate
    a higher number of exoplanets, but would definitely not recommend.


    Args:
        exoData ([type]): Degrees of pointing and number of visible exoplanets dataframe
    """    
    linearReg = lm.LinearRegression()
    linearReg.fit(exoData.iloc[:,0], exoData.iloc[:,1])

    coefficients = linearReg.coef_

    return coefficients


start = datetime.now()

data = pd.read_csv(fileName, sep=',', skiprows=38)
exo_data = data[['ra', 'dec', 'pl_name']]

unconfirmed_data = pd.read_csv(unconfirmedFileName, sep=',', skiprows=50)
unconfirmed_exo_data = unconfirmed_data[['ra','dec','pl_name']]

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

raBasic = np.linspace(0, raResolution, raResolution)
decMoon = teleview(raBasic)

raBasic[np.where(decMoon < -90)] = raBasic[np.where(decMoon < -90)] + 180
decMoon[np.where(decMoon < -90)] = -decMoon[np.where(decMoon < -90)] - 180

raBasic[np.where(raBasic > 360)] = raBasic[np.where(raBasic > 360)] - 360
raBasic[np.where(raBasic < 0)] = raBasic[np.where(raBasic < 0)] + 360

rTelescope = anglesToVector(raBasic, decMoon)
rExos = anglesToVector(exo_ra, exo_dec).T
rUnconfirmedExos = anglesToVector(unconfirmed_exo_ra,unconfirmed_exo_dec).T


# 3D Projection?
'''rTelescope2 = rTelescope.T
print(rTelescope2[0])
origin = np.shape(rTelescope2)
print('Origin:')
print(origin)

ax = plt.figure().add_subplot(projection='3d')
ax.quiver(rTelescope2[0], rTelescope2[1], rTelescope2[2], origin[0], origin[1], origin[2])
plt.show()
# print(f"rTelescope = {rTelescope[4]}")
# print(f"rExos = {rExos.T[1]}")'''


#Confirmed exoplanets

gamma = getGamma(rTelescope, rExos)
gammaMin = pd.DataFrame([np.min(gamma, axis=0)], columns=exo_names)
gammaMax = pd.DataFrame([np.max(gamma, axis=0)], columns=exo_names)
gammaDiff = pd.DataFrame([np.max(gamma, axis=0)- np.min(gamma, axis=0)], columns=exo_names)


gammaDF = pd.DataFrame(gamma, columns=exo_names)


#Unconfirmed exoplanets

unconfirmedGamma = getGamma(rTelescope,rUnconfirmedExos)
unconfirmedGammaMin = pd.DataFrame([np.min(unconfirmedGamma, axis=0)], columns=unconfirmed_exo_names)
unconfirmedGammaMax = pd.DataFrame([np.max(unconfirmedGamma, axis=0)], columns=unconfirmed_exo_names)
unconfirmedGammaDiff = pd.DataFrame([np.max(unconfirmedGamma, axis=0)- np.min(unconfirmedGamma, axis=0)], columns=unconfirmed_exo_names)

unconfirmedGammaDF = pd.DataFrame(unconfirmedGamma, columns=unconfirmed_exo_names)



def thingy(qquantile, colorBar, DF:pd.DataFrame, MinDF:pd.DataFrame,MaxDF:pd.DataFrame,DiffDF:pd.DataFrame):
    """Thingy calculates the number of visible exoplanets as a function of pointing angle for different quantiles. 

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



for quantile, color in zip(quantiles, colours):
    thingy(quantile, color, gammaDF, gammaMin, gammaMax, gammaDiff)



#exos = pd.read_csv(f"data/0.223 Numbers.csv")
#print(exos)

#reg_coef = regressionHammer(exos)
#print(reg_coef)

'''Uncomment for confirmed and unconfirmed exoplanets'''

#for quantile, colour in zip(quantiles, unconfirmed_colours):
    #thingy(quantile, colour, unconfirmedGammaDF, unconfirmedGammaMin, unconfirmedGammaMax, unconfirmedGammaDiff)



print(f"Runtime = {datetime.now()-start}")

plt.legend()
plt.title('Confirmed Exoplanets in sight as a function of pointing angle')
plt.xlabel('Pointing angle [°]')
plt.ylabel('Number of visible exoplanets')
plt.grid(which='both', axis='y')
plt.show()

'''Uncomment for confirmed and unconfirmed exoplanets - WARNING: I am not responsible by any damage to your retinas caused by the colour scheme ~N'''


""" for quantile, colour in zip(quantiles, unconfirmed_colours):
    thingy(quantile, colour, unconfirmedGammaDF, unconfirmedGammaMin, unconfirmedGammaMax, unconfirmedGammaDiff)


plt.legend()
plt.title('Confirmed and Unconfirmed Exoplanets in sight as a function of pointing angle')
plt.xlabel('Pointing angle [°]')
plt.ylabel('Number of visible exoplanets')
plt.grid(which='both', axis='y')
plt.show()
 """

