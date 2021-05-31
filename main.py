import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import pyquaternion as qt
from tqdm import tqdm as tqdm
from datetime import datetime
from math import cos, sin, asin, acos, atan2, tan
from math import radians as mrad
from numba import njit, jit


fileName = "Data_new_encoding.csv"

barReso = 5
raResolution = 360
colours = ["blue", "orange", "green", "red", 'purple']

ra_of_asc_node = 125.08
craterLatitude = -88.5
craterLongitude = 152.0
A = 16.56

moonOrbitalPeriod = 28 # Days
hrsPerExo = 150 # Hours
quantileExo = hrsPerExo/(moonOrbitalPeriod*24)
print(f"Exoplanet quantile = {quantileExo}")


def teleview(raBasic: np.ndarray, lat: float = craterLatitude, lon: float = craterLongitude, A: float = A, an: float = ra_of_asc_node) -> np.ndarray:
    """get declination of given position in array form

    Args:
        craterLatitude (float): crater latitude
        A (float): Amplitude moon tilt
        ra_basic (np.ndarray): right ascension
        an (float): right ascension of ascending node

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


start = datetime.now()

data = pd.read_csv(fileName, sep=',', skiprows=38)
exo_data = data[['ra', 'dec', 'pl_name']]

print(f"Total exoplanets = {len(exo_data)}")
print(f"Total exoplanets below 18 dec = {len(exo_data[exo_data['dec'] < 18])}")

# exo_data = exo_data[exo_data.dec <= 3]
exo_ra = exo_data.iloc[:, 0].to_numpy()
exo_dec = exo_data.iloc[:, 1].to_numpy()
exo_names = exo_data.iloc[:, 2].to_list()

raBasic = np.linspace(0, raResolution, raResolution)
decMoon = teleview(raBasic)

raBasic[np.where(decMoon < -90)] = raBasic[np.where(decMoon < -90)] + 180
decMoon[np.where(decMoon < -90)] = -decMoon[np.where(decMoon < -90)] - 180

raBasic[np.where(raBasic > 360)] = raBasic[np.where(raBasic > 360)] - 360
raBasic[np.where(raBasic < 0)] = raBasic[np.where(raBasic < 0)] + 360

rTelescope = anglesToVector(raBasic, decMoon)
rExos = anglesToVector(exo_ra, exo_dec).T



# 3D Projection stuff?
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


#----------------Temporarily out of commission------------------
gamma = getGamma(rTelescope, rExos)
gammaMin = pd.DataFrame([np.min(gamma, axis=0)], columns=exo_names)
gammaMax = pd.DataFrame([np.max(gamma, axis=0)], columns=exo_names)
gammaDiff = pd.DataFrame([np.max(gamma, axis=0)- np.min(gamma, axis=0)], columns=exo_names)
#gammeAverage = pd.DataFrame([np.])
# print(f"gamma = {gamma[4, 1]}")

gammaDF = pd.DataFrame(gamma, columns=exo_names)

'''gamma = getGamma(rTelescope, rExos)
gammaDF = pd.DataFrame(gamma, columns=exo_names)
gammaMin = gammaDF.min(axis=0)
gammaMax = gammaDF.max(axis=0)
gammaDiff = gammaMax-gammaMin'''




# print(gammaDF)

def thingy(qquantile, colorBar):
    exoQuantile = pd.DataFrame(gammaDF.quantile(qquantile, axis=0).rename("Quantile")).transpose()
    exoQuantile = exoQuantile.append(gammaMin.iloc[0].rename("Min"), ignore_index=False)
    exoQuantile = exoQuantile.append(gammaMax.iloc[0].rename("Max"), ignore_index=False)
    exoQuantile = exoQuantile.append(gammaDiff.iloc[0].rename("Diff"), ignore_index=False)
    #exoQuantile = exoQuantile.append()

    for angle in [x*barReso for x in range(0,(int(90/barReso) + 1))]:
        exoQuantile = exoQuantile.append(exoQuantile.loc["Quantile"].rename(f"{angle}") <= angle, ignore_index=False)

    sumVisible = ((exoQuantile == 1).astype(int).sum(axis=1)).iloc[4::]

    exoQuantile.to_csv(f"data/{qquantile}.csv")
    plt.bar(sumVisible.index, sumVisible.values, color=colorBar, label=f"q = {quantile}")

quantiles = [0, 0.25, 0.5, 0.75, 1]

for quantile, color in zip(quantiles, colours):
    thingy(quantile, color)



print(f"Runtime = {datetime.now()-start}")

plt.legend()
plt.show()

