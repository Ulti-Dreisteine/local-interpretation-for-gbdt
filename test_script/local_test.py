# -*- coding: utf-8 -*-
"""
Created on 2020/8/5 5:55 下午

@File: build_local_test_samples.py

@Department: AI Lab, Rockontrol, Chengdu

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 构建本地测试样本
"""

import pandas as pd
import numpy as np
import sys, os

sys.path.append('../')

from src import proj_dir


def _fill_nans(data: pd.DataFrame) -> pd.DataFrame:
	"""数据缺失值填补"""
	data = data.copy()
	values = {}
	for col in data.columns:
		values.update({col: data[col].mean()})
	data.fillna(values, inplace = True)
	return data


def build_samples():
	"""构建样本"""
	X_train = pd.read_csv(os.path.join(proj_dir, 'data/raw/X_train.csv'))
	y_train = pd.read_csv(os.path.join(proj_dir, 'data/raw/y_train.csv'))
	
	# 特征名.
	features = list(X_train.columns)
	
	# 数据缺失值填补.
	X_train, y_train = _fill_nans(X_train), _fill_nans(y_train)
	
	# 转换为样本格式.
	X_train = np.array(X_train)
	y_train = np.array(y_train).flatten()
	
	return X_train, y_train, features


# ---- 构建数据集 -----------------------------------------------------------------------------------

X_train, y_train, features = build_samples()
loc_id = 20
x_test = X_train[loc_id, :].reshape(1, -1)



