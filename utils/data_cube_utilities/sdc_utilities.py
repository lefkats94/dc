# Copyright 2018 GRID-Geneva. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

import numpy as np
import xarray as xr

def create_scl_clean_mask(scl, valid_cats):
    """
    Description:
      Create a clean mask from a list of valid categories,
    Input:
      scl (xarray) - scl from dc_preproc product (generated with sen2cor)
    Args:
      scl: xarray data array to extract clean categories from.
      valid_cats: array of ints representing what category should be considered valid.
    Output:
      clean_mask (boolean numpy array)
    """

    ###################################
    # scl values:                     #
    #   0 - no data                   #
    #   1 - saturated or defective    #
    #   2 - dark area pixels          #
    #   3 - cloud_shadows             #
    #   4 - vegetation                #
    #   5 - not vegetated             #
    #   6 - water                     #
    #   7 - unclassified              #
    #   8 - cloud medium probability  #
    #   9 - cloud high probability    #
    #  10 - thin cirrus               #
    #  11 - snow                      #
    ###################################

    return xr.apply_ufunc(np.isin, scl, valid_cats).values
