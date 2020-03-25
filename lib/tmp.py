# -*- coding: utf-8 -*-
"""
Created on 2020/3/25 14:01

@Project -> File: local-interpretation-for-gbdt -> tmp.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe:
"""

import logging

logging.basicConfig(level = logging.INFO)

import pandas as pd
import numpy as np
import sys, os

sys.path.append('../')

from lib import proj_dir


def _fill_nans(data: pd.DataFrame) -> pd.DataFrame:
	data = data.copy()
	
	values = {}
	for col in data.columns:
		values.update({col: data[col].mean()})
	
	data.fillna(values, inplace = True)
	return data


def build_samples():
	# %% 载入数据和处理.
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


X_train, y_train, features = build_samples()

# 正样本.
pos_id = 60
x_test = X_train[pos_id, :].reshape(1, -1)



