"""
This script calculates the autocorrelation of vector v and vector delta_v
and performs dickey-fuller test (on training data)
"""

import os

import numpy as np
from statsmodels.tsa.stattools import adfuller

import convert_coord as coord

def acf(time_sequence, length=101):
    """Statistical autocorrelation function"""
    return np.array([1]+[np.corrcoef(time_sequence[:-i], time_sequence[i:])[0, 1]
                         for i in range(1, length)])


PATH_TO_DATA = "/media/demo/DATA/saliency-exploitation/sw/preprocessed_train"
FILE_NAMES = os.listdir(PATH_TO_DATA)

ACF_LENGTH = 100 # 100 * 10ms = 1 second

N = len(FILE_NAMES)
VIDEO_WIDTH = 3840

autocorrelation = np.zeros((N, 3, ACF_LENGTH+1))

autocorrelation_delta = np.zeros((N, 3, ACF_LENGTH+1))

adf_stats = np.zeros((N, 3, 2))
adf_stats_delta = np.zeros((N, 3, 2))

# Loop through each video
for i, file_name in enumerate(FILE_NAMES):

    data = np.load(os.path.join(PATH_TO_DATA, file_name))

    video_height = 2 * data[0, 1]

    x, y, z = coord.equirect_to_cart(data[:, 2:4], VIDEO_WIDTH, video_height)

    delta_x = x[1:] - x[:-1]
    delta_y = y[1:] - y[:-1]
    delta_z = z[1:] - z[:-1]

    autocorrelation[i, 0] = acf(x, length=ACF_LENGTH+1)
    autocorrelation[i, 1] = acf(y, length=ACF_LENGTH+1)
    autocorrelation[i, 2] = acf(z, length=ACF_LENGTH+1)

    autocorrelation_delta[i, 0] = acf(delta_x, length=ACF_LENGTH+1)
    autocorrelation_delta[i, 1] = acf(delta_y, length=ACF_LENGTH+1)
    autocorrelation_delta[i, 2] = acf(delta_z, length=ACF_LENGTH+1)

    adf_stats[i, 0, 0:2] = adfuller(x)[0:2]
    adf_stats[i, 1, 0:2] = adfuller(y)[0:2]
    adf_stats[i, 2, 0:2] = adfuller(z)[0:2]

    adf_stats_delta[i, 0, 0:2] = adfuller(delta_x)[0:2]
    adf_stats_delta[i, 1, 0:2] = adfuller(delta_y)[0:2]
    adf_stats_delta[i, 2, 0:2] = adfuller(delta_z)[0:2]
    print(i, N)

delays_autocorr = np.mean(
    autocorrelation[:, :, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]], axis=0)

DELAYS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
COORDS = ["x", "y", "z"]

print("Autocorrelation")
for i in range(3):
    print(COORDS[i])
    for j in range(10):
        print("(" + str(DELAYS[j])+", " + str(delays_autocorr[i, j]) + ")")

print("\n")
print("Autocorrelation differences")
delays_autocorr_delta = np.mean(
    autocorrelation_delta[:, :, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]], axis=0)

for i in range(3):
    print(COORDS[i])
    for j in range(10):
        print("(" + str(DELAYS[j])+", " +
              str(delays_autocorr_delta[i, j]) + ")")

print("\n\n")
print("Dickey-Fuller test")
for i in range(3):
    print(COORDS[i])
    print('ADF Statistic: %f' % np.mean(adf_stats, axis=0)[i, 0])
    print('p-value: %f' % np.mean(adf_stats, axis=0)[i, 1])

print("\n")

print("Dickey-Fuller test differences")

for i in range(3):
    print(COORDS[i])
    print('ADF Statistic: %f' % np.mean(adf_stats_delta, axis=0)[i, 0])
    print('p-value: %f' % np.mean(adf_stats_delta, axis=0)[i, 1])


# Autocorrelation
# x
# (0.1, 0.9938891659097956)
# (0.2, 0.9817185922445864)
# (0.3, 0.964052015571952)
# (0.4, 0.9423665012227856)
# (0.5, 0.917694083101744)
# (0.6, 0.8908774618838725)
# (0.7, 0.8625920628246342)
# (0.8, 0.8333748902610281)
# (0.9, 0.8037171060756304)
# (1, 0.7739643013732441)
# y
# (0.1, 0.9916017625634447)
# (0.2, 0.9691874831465157)
# (0.3, 0.9370585019088727)
# (0.4, 0.8984273569331335)
# (0.5, 0.8556208193614915)
# (0.6, 0.8104236908007973)
# (0.7, 0.764252507957794)
# (0.8, 0.718171479412073)
# (0.9, 0.6729410914631531)
# (1, 0.6291108688071693)
# z
# (0.1, 0.9945224897386908)
# (0.2, 0.982433991910164)
# (0.3, 0.9648336916022833)
# (0.4, 0.9431034766391365)
# (0.5, 0.9182898059802391)
# (0.6, 0.8912025376092509)
# (0.7, 0.8624646009011945)
# (0.8, 0.8326294593690106)
# (0.9, 0.8021254376803344)
# (1, 0.7713385556330352)


# Autocorrelation differences
# x
# (0.1, 0.39341273188258485)
# (0.2, 0.31322569456706684)
# (0.3, 0.24210646108579462)
# (0.4, 0.1860224339406125)
# (0.5, 0.14099168882175706)
# (0.6, 0.10508016303750924)
# (0.7, 0.07518767572510962)
# (0.8, 0.04827858974940249)
# (0.9, 0.027743491819751158)
# (1, 0.011696885736396195)
# y
# (0.1, 0.8199353909717725)
# (0.2, 0.5907322344979022)
# (0.3, 0.4248540177817428)
# (0.4, 0.30319873187847324)
# (0.5, 0.20417661221823588)
# (0.6, 0.1207115656710825)
# (0.7, 0.05554977873096029)
# (0.8, 0.00694656305164899)
# (0.9, -0.030414438090112128)
# (1, -0.05827036528783815)
# z
# (0.1, 0.405280185430189)
# (0.2, 0.323849862494225)
# (0.3, 0.24940537336899588)
# (0.4, 0.19244793179108483)
# (0.5, 0.14806536212678184)
# (0.6, 0.11229770876334129)
# (0.7, 0.08415198614661709)
# (0.8, 0.059813583303891735)
# (0.9, 0.03978558360523195)
# (1, 0.022932190056888714)



# Dickey-Fuller test
# x
# ADF Statistic: -3.358430
# p-value: 0.079566
# y
# ADF Statistic: -4.155788
# p-value: 0.019260
# z
# ADF Statistic: -3.280729
# p-value: 0.090958


# Dickey-Fuller test differences
# x
# ADF Statistic: -8.817093
# p-value: 0.000666
# y
# ADF Statistic: -8.147880
# p-value: 0.000003
# z
# ADF Statistic: -8.664221
# p-value: 0.000178
