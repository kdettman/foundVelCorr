import pandas as pd
import numpy as np

def cosmo_cuts(data, c_cut=0.3, x1_cut=3.0, z_cut=0.008):
    data_temp = data.copy(deep=True)
    data_temp = data_temp[np.abs(data_temp.c) < c_cut]
    data_temp = data_temp[np.abs(data_temp.x1) < x1_cut]
    data_temp = data_temp[data_temp.z > z_cut]
    return data_temp

def err_cuts(data, c_err_cut=0.5, x1_err_cut=1.0, shp_cor_cut=1.0, vel_err_cut=1.0, alt_cerr_key=None,alt_x1err_key=None,alt_shpcor_key=None,alt_velerr_key=None):
	data_temp = data.copy(deep=True)
	if c_err_cut is not None:
		if alt_cerr_key is None:
			data_temp = data_temp[data_temp.c_err < c_err_cut]
		else:
			data_temp = data_temp[data_temp[alt_cerr_key] < c_err_cut]
	if x1_err_cut is not None:
		if alt_x1err_key is None:
			data_temp = data_temp[data_temp.x1_err < x1_err_cut]
		else:
			data_temp = data_temp[data_temp[alt_x1err_key] < x1_err_cut]
	if shp_cor_cut is not None:
		if alt_shpcor_key is None:
			data_temp = data_temp[data_temp.shp_corr_err < shp_cor_err_cut]
		else:
			data_temp = data_temp[data_temp[alt_shpcor_key] < shp_cor_err_cut]
	if vel_err_cut is not None:
		if alt_velerr_key is None:
			data_temp = data_temp[data_temp.velerr < vel_err_cut]
		else:
			data_temp = data_temp[data_temp[alt_velerr_key] < vel_err_cut]
	return data_temp