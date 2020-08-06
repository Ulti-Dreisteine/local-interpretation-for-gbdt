# -*- coding: utf-8 -*-
"""
Created on 2020/8/5 5:25 下午

@File: local_interp_for_gbdt_new.py

@Department: AI Lab, Rockontrol, Chengdu

@Author: luolei

@Email: dreisteine262@163.com

@Describe: GBDT的局部解释模型
"""

import logging

logging.basicConfig(level = logging.INFO)

from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np
import sys

sys.path.append('../')

from src.decision_tree_info import DecisionTreeInfo


def train_model(X_train: np.ndarray, y_train: np.ndarray, **params) -> (GradientBoostingClassifier, np.ndarray):
	"""训练模型并计算各子模型权重"""
	# 初始化并训练模型.
	clf = GradientBoostingClassifier(**params)
	clf.fit(X_train, y_train)
	
	# 计算各子模型的损失函数值, 即train_score, 参见GradientBoostingClassifier.__doc__.
	# 按照我的理解, 各子模型损失函数值可以作为对应子模型的权重用于后续分数加权, 参见文
	# 献Table 1中相关内容.
	tree_weights = clf.train_score_
	
	return clf, tree_weights


class LocalInterpForGBDT(object):
	"""GBDT局部解释器"""
	
	def __init__(self, X_train: np.ndarray, y_train: np.ndarray):
		self.X_train, self.y_train = X_train.copy(), y_train.copy()
	
	def train_model(self, **params):
		"""
		训练模型
		:param params: 参见sklearn.ensemble.GradientBoostingClassifier.__doc__
		"""
		self.clf, self.tree_weights = train_model(self.X_train, self.y_train, **params)
		
	def _cal_FC_for_all_trees(self, x_test) -> dict:
		FC_results = {}
		for i in range(self.clf.n_estimators):
			tree_ = self.clf.estimators_[i, 0]
			dti = DecisionTreeInfo(tree_)
			nodes_labels_counts, nodes_labels_ratio = dti.cal_nodes_labels_counts_ratio(
				X_train, y_train
			)
			FC = dti.cal_FC_for_sample(x_test, nodes_labels_counts, nodes_labels_ratio)
			FC_results[i] = FC
		return FC_results
	
	def local_interpretation(self, x_test: np.ndarray, features: list, top_n: int = None) -> (list, list):
		# 计算每棵树上的FC值.
		FC_results = self._cal_FC_for_all_trees(x_test)
		
		# 合并FC结果.
		FC_df = pd.DataFrame.from_dict(FC_results, orient = 'index').sort_index()
		FC_merged = pd.DataFrame(
			np.dot(
				self.tree_weights.reshape(1, -1), FC_df.fillna(0.0)
			),
			columns = FC_df.columns
		).T
		FC_merged.columns = ['FC_value']
		FC_merged['feature_id'] = FC_merged.index
		FC_merged.sort_values('FC_value', ascending = False, inplace = True)
		FC_merged.reset_index(drop = True, inplace = True)
		
		if top_n is not None:
			FC_merged = FC_merged.iloc[: top_n][:]
		
		# 总结结果.
		top_features = [features[p] for p in list(FC_merged['feature_id'])]
		top_FC_values = list(FC_merged['FC_value'])
		
		return top_features, top_FC_values
	
	
if __name__ == '__main__':
	import matplotlib.pyplot as plt
	
	# ---- 载入数据和处理 ---------------------------------------------------------------------------
	
	from test_script.local_test import X_train, y_train, features, x_test
	
	# ---- 初始化对象 -------------------------------------------------------------------------------
	
	self = LocalInterpForGBDT(X_train, y_train)
	
	# ---- 训练模型 ---------------------------------------------------------------------------------
	
	params = {
		'n_estimators': 200,
		'learning_rate': 0.1,
		'min_samples_split': 10,
		'min_samples_leaf': 2,
		'max_depth': 3,
		'subsample': 1,
		'random_state': 10,
		'loss': 'deviance',
	}
	self.train_model(**params)
	
	FC_results = self._cal_FC_for_all_trees(x_test)
	top_features, top_FC_values = self.local_interpretation(x_test, features)
	
	# 画图.
	plt.figure(figsize = [8, 6])
	plt.bar(top_features, top_FC_values)
	plt.xticks(range(len(top_features)), top_features, rotation = 90, fontsize = 6)
	plt.yticks(fontsize = 6)
	plt.ylabel('feature importance score', fontsize = 10.0)
	plt.tight_layout()
	
	
	
	


