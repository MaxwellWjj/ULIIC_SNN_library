"""
Author: Jiajun Wu, HUST, China
Main Library: Pytorch
Description: It is a library with spiking neural network. We would like to implement this NN in hardware.
File Information: this file includes exp computing methods, such as CORDIC.
Log: 2020/1/20 Build firstly
Reference: Bindsnet library https://bindsnet-docs.readthedocs.io/

"""

import torch
import math
import numpy as np


def error_exp(x, error=True, bits_width=16, mode=0, cordic_mode=3):
    """

    :param x: The value to be computed.
    :param error: Whether we need the error
    :param bits_width: The CORDIC width, 8, 16, 24, 32, default value is 8.
    :param mode: Average mode or Worst mode, default is average.
    :param cordic_mode: cordic_mode, 0-control, 1-conventional, 2-angle recoding, 3-pipeline.

    """

    exp_result = 0.0
    if error == False:  # accurate result
        exp_result = torch.exp(x)
        return exp_result

    if mode == 0 and error:   # average
        if bits_width == 8:
            if cordic_mode == 0:
                exp_result = torch.exp(x) + 0.01557
            if cordic_mode == 1:
                exp_result = torch.exp(x) + 0.01654
            if cordic_mode == 2:
                exp_result = torch.exp(x) + 0.010595
            if cordic_mode == 3:
                exp_result = torch.exp(x) + 0.02143
        if bits_width == 16:
            if cordic_mode == 0:
                exp_result = torch.exp(x) + 1.74e-4
            if cordic_mode == 1:
                exp_result = torch.exp(x) + 9.0429e-5
            if cordic_mode == 2:
                exp_result = torch.exp(x) + 0.000127197
            if cordic_mode == 3:
                exp_result = torch.exp(x) + 0.00014967
        if bits_width == 24:
            if cordic_mode == 0:
                exp_result = torch.exp(x) + 1.18e-6
            if cordic_mode == 1:
                exp_result = torch.exp(x) + 4.65562e-7
            if cordic_mode == 2:
                exp_result = torch.exp(x) + 8.27986e-7
            if cordic_mode == 3:
                exp_result = torch.exp(x) + 1.19239e-6
        if bits_width == 32:
            if cordic_mode == 0:
                exp_result = torch.exp(x) + 7.95e-9
            if cordic_mode == 1:
                exp_result = torch.exp(x) + 2.12247e-9
            if cordic_mode == 2:
                exp_result = torch.exp(x) + 6.27273e-9
            if cordic_mode == 3:
                exp_result = torch.exp(x) + 7.94588e-9

    if mode == 1 and error:   # average
        if bits_width == 8:
            if cordic_mode == 0:
                exp_result = torch.exp(x) + 0.04866
            if cordic_mode == 1:
                exp_result = torch.exp(x) + 0.06592
            if cordic_mode == 2:
                exp_result = torch.exp(x) + 0.0445
            if cordic_mode == 3:
                exp_result = torch.exp(x) + 0.06258
        if bits_width == 16:
            if cordic_mode == 0:
                exp_result = torch.exp(x) + 6.12e-4
            if cordic_mode == 1:
                exp_result = torch.exp(x) + 0.000291189
            if cordic_mode == 2:
                exp_result = torch.exp(x) + 0.000504581
            if cordic_mode == 3:
                exp_result = torch.exp(x) + 0.000504581
        if bits_width == 24:
            if cordic_mode == 0:
                exp_result = torch.exp(x) + 3.69e-6
            if cordic_mode == 1:
                exp_result = torch.exp(x) + 1.6199e-6
            if cordic_mode == 2:
                exp_result = torch.exp(x) + 2.61432e-6
            if cordic_mode == 3:
                exp_result = torch.exp(x) + 2.90097e-6
        if bits_width == 32:
            if cordic_mode == 0:
                exp_result = torch.exp(x) + 2.38e-8
            if cordic_mode == 1:
                exp_result = torch.exp(x) + 8.25555e-9
            if cordic_mode == 2:
                exp_result = torch.exp(x) + 1.76028e-8
            if cordic_mode == 3:
                exp_result = torch.exp(x) + 2.07546e-8

    if mode == 2 and error:
        exp_result = torch.exp(x) + 1e-10

    # print(torch.exp(x), exp_result)
    return exp_result


def conventional_cordic_exp(z: torch.Tensor, error=True, bits_width=16):
    # y = 0
    # x = scale
    y = torch.zeros(z.shape)
    x = torch.zeros(z.shape)

    if bits_width == 8:
        x += 1.2109375
    if bits_width == 16:
        x += 1.207489013671875
    if bits_width == 24:
        x += 1.207497000694275
    if bits_width == 32:
        x += 1.2074970677495

    if error == False:
        exp_result = torch.exp(z)
        return exp_result

    list_thetai = [0.549306144334055, 0.255412811882995, 0.125657214140453, 0.0625815714770030,
                   0.0312601784906670, 0.0156262717520522, 0.00781265895154042, 0.00390626986839683,
                   0.00195312748353255, 0.000976562810441036, 0.000488281288805113, 0.000244140629850639,
                   0.000122070313106330, 6.10351563257912e-05, 3.05175781344739e-05, 1.52587890636842e-05,
                   7.62939453139803e-06, 3.81469726564350e-06, 1.90734863281481e-06, 9.53674316406539e-07,
                   4.76837158203161e-07, 2.38418579101567e-07, 1.19209289550782e-07, 5.96046447753907e-08,
                   2.98023223876953e-08, 1.49011611938477e-08, 7.45058059692383e-09, 3.72529029846191e-09,
                   1.86264514923096e-09, 9.31322574615479e-10, 4.65661287307739e-10, 2.32830643653870e-10]

    for i in range(bits_width):

        if i == 3 or i == 12:
            # if z > 0:
            #     x += y * math.pow(2, -(i+1))
            #     y += (x - y * math.pow(2, -(i+1))) * math.pow(2, -(i+1))
            #     z -= list_thetai[i]
            #     print("z>0", z, i)
            #
            # else:
            #     x -= y * math.pow(2, -(i+1))
            #     y -= (x + y * math.pow(2, -(i+1))) * math.pow(2, -(i+1))
            #     z += list_thetai[i]
            #     print("z<0", z, i)

            signz = z.sign()
            x += y * signz * math.pow(2, -(i+1))
            y += (x - y * signz *math.pow(2, -(i+1))) * signz * math.pow(2, -(i+1))
            z -= signz * list_thetai[i]
            # print("z>0", z, i)

        # if z > 0:
        #     x += y * math.pow(2, -(i+1))
        #     y += (x - y * math.pow(2, -(i+1))) * math.pow(2, -(i+1))
        #     z -= list_thetai[i]
        #     print("z>0", z, i)
        #
        # else:
        #     x -= y * math.pow(2, -(i+1))
        #     y -= (x + y * math.pow(2, -(i+1))) * math.pow(2, -(i+1))
        #     z += list_thetai[i]
        #     print("z<0", z, i)

        signz = z.sign()
        x += y * signz * math.pow(2, -(i + 1))
        y += (x - y * signz * math.pow(2, -(i + 1))) * signz * math.pow(2, -(i + 1))
        z -= signz * list_thetai[i]
        # print("z>0", z, i)

    exp_result = x + y
    return exp_result


def pipeline_cordic_exp(z: torch.Tensor, error=True, bits_width=16):
    # y = 0
    # x = scale
    y = torch.zeros(z.shape)
    x = torch.zeros(z.shape)
    list_recode = torch.zeros(32)

    if bits_width == 8:
        x += 1
    if bits_width == 16:
        x += 1
    if bits_width == 24:
        x += 1
    if bits_width == 32:
        x += 1

    if error == False:
        exp_result = torch.exp(z)
        return exp_result

    list_thetai = [0.549306144334055, 0.255412811882995, 0.125657214140453, 0.0625815714770030,
                   0.0312601784906670, 0.0156262717520522, 0.00781265895154042, 0.00390626986839683,
                   0.00195312748353255, 0.000976562810441036, 0.000488281288805113, 0.000244140629850639,
                   0.000122070313106330, 6.10351563257912e-05, 3.05175781344739e-05, 1.52587890636842e-05,
                   7.62939453139803e-06, 3.81469726564350e-06, 1.90734863281481e-06, 9.53674316406539e-07,
                   4.76837158203161e-07, 2.38418579101567e-07, 1.19209289550782e-07, 5.96046447753907e-08,
                   2.98023223876953e-08, 1.49011611938477e-08, 7.45058059692383e-09, 3.72529029846191e-09,
                   1.86264514923096e-09, 9.31322574615479e-10, 4.65661287307739e-10, 2.32830643653870e-10]

    list_scale = [1.15470053837925, 1.03279555898864, 1.00790526135794, 1.00195886573624,
                  1.00048863916916, 1.00012209266879, 1.00003051897518, 1.00000762948184,
                  1.00000190735409, 1.00000047683750, 1.00000011920931, 1.00000002980232,
                  1.00000000745058, 1.00000000186265, 1.00000000046566, 1.00000000011642,
                  1.00000000002910, 1.00000000000728, 1.00000000000182, 1.00000000000045,
                  1.00000000000011, 1.00000000000003, 1.00000000000001, 1.00000000000000,
                  1.00000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000,
                  1.00000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000]

    for i in range(bits_width):
        z_temp = z.sign() * z
        for m in range(z.shape[0]):
            for n in range(z.shape[1]):
                if z_temp[m][n] > math.pow(2, -(bits_width - 1)):
                    for j in range(bits_width):
                        if z_temp[m][n] > list_thetai[j]:
                            list_recode[j] = z_temp[m][n] - list_thetai[j]
                        else:
                            list_recode[j] = list_thetai[j] - z_temp[m][n]

                    rmin = list_recode[bits_width - 1]
                    irecode = bits_width - 1

                    for k in range(bits_width - 2):
                        l = bits_width - (k + 2)
                        if list_recode[l] < rmin:
                            rmin = list_recode[l]
                            irecode = l
                        else:
                            pass

                    signz = z.sign()
                    x[m][n] += y[m][n] * signz[m][n] * math.pow(2, -(irecode + 1))
                    y[m][n] += (x[m][n] - y[m][n] * signz[m][n] * math.pow(2, -(irecode + 1))) * \
                               signz[m][n] * math.pow(2, -(irecode + 1))
                    z[m][n] -= signz[m][n] * list_thetai[irecode]
                    x[m][n] = x[m][n] * list_scale[irecode]
                    y[m][n] = y[m][n] * list_scale[irecode]
                    # print("z>0", z, i)
                else:
                    break

    # for i in range(bits_width):
    #     z_temp = z.sign() * z
    #     if z_temp > math.pow(2, -(bits_width - 1)):
    #         for j in range(bits_width):
    #             if z_temp > list_thetai[j]:
    #                 list_recode[j] = z_temp - list_thetai[j]
    #             else:
    #                 list_recode[j] = list_thetai[j] - z_temp
    #
    #         rmin = list_recode[bits_width - 1]
    #         irecode = bits_width - 1
    #
    #         for k in range(bits_width - 2):
    #             l = bits_width - (k + 2)
    #             if list_recode[l] < rmin:
    #                 rmin = list_recode[l]
    #                 irecode = l
    #             else:
    #                 pass
    #
    #         signz = z.sign()
    #         x += y * signz * math.pow(2, -(irecode + 1))
    #         y += (x - y * signz * math.pow(2, -(irecode + 1))) * signz * math.pow(2, -(irecode + 1))
    #         z -= signz * list_thetai[irecode]
    #         x = x * list_scale[irecode]
    #         y = y * list_scale[irecode]
    #         # print("z>0", z, i)
    #     else:
    #         break

    exp_result = x + y
    return exp_result


if __name__ == '__main__':
    z = torch.zeros(1)
    z += 0.8
    print(z)
    # true_res = torch.exp(z)
    # cordic_res = error_exp(z)
    true_res = torch.exp(z)
    cordic_res = pipeline_cordic_exp(z)
    print(z)
    print(true_res)
    print(cordic_res)
    print(cordic_res - true_res)
