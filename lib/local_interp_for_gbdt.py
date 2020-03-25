# -*- coding: utf-8 -*-
"""
Created on 2020/3/25 11:26

@Project -> File: local-interpretation-for-gbdt -> local_interp_for_gbdt.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe:
"""

import logging

logging.basicConfig(level = logging.INFO)

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('../')


def _train_gbdt(X_train, y_train, params: dict) -> (GradientBoostingClassifier, np.ndarray):
	gbdt = GradientBoostingClassifier(
		**params
	)
	gbdt.fit(X_train, y_train)
	
	# 计算每棵树的权重.
	tree_weights = []
	for i in range(gbdt.n_estimators):
		tree_ = gbdt[i, 0]
		y_pred = tree_.predict(X_train)
		s = np.sum(np.power(y_pred, 2))
		tree_weights.append(s)
	tree_weights = np.array(tree_weights) / np.sum(tree_weights)
	
	return gbdt, tree_weights


class DecisionTreeInfo(object):
	"""
	决策树信息计算
	"""
	
	def __init__(self, tree: DecisionTreeRegressor):
		self.tree = tree
		self.node_count = tree.tree_.node_count
	
	def _get_nodes_connects(self) -> dict:
		"""
		获取父子节点连接关系, key为父节点编号, value分别对应左右分支子节点编号, -1代表没有子节点
		"""
		nodes_connects = {}
		for i in range(self.node_count):
			nodes_connects.update({i: {'left': self.tree.tree_.children_left[i], 'right': self.tree.tree_.children_right[i]}})
		return nodes_connects
	
	@property
	def nodes_connects(self):
		return self._get_nodes_connects()
	
	def _count_nodes_samples_number(self) -> dict:
		nodes_samples_n = {}
		for i in range(self.node_count):
			nodes_samples_n.update({i: self.tree.tree_.n_node_samples[i]})
		return nodes_samples_n
	
	@property
	def nodes_samples_number(self):
		return self._count_nodes_samples_number()
	
	def _get_nodes_split_thres(self):
		nodes_split_thres = {}
		for i in range(self.node_count):
			nodes_split_thres.update({i: self.tree.tree_.threshold[i]})
		return nodes_split_thres
	
	@property
	def nodes_split_thres(self):
		return self._get_nodes_split_thres()
	
	def _get_nodes_split_features(self):
		nodes_split_features = {}
		for i in range(self.node_count):
			nodes_split_features.update({i: self.tree.tree_.feature[i]})
		return nodes_split_features
	
	@property
	def nodes_split_features(self):
		return self._get_nodes_split_features()
	
	def cal_nodes_pos_neg_counts_ratio(self, X_train: np.ndarray, y_train: np.ndarray):
		"""
		获取各节点上正负样本比例
		
		Notes:
		------------------------------------------------------------
		目前这一步还需要X_train, y_train正向计算, 按理说当训练tree的时候就应该
		能获得节点上的正负样本信息, 后续需要改进这一步计算.
		"""
		try:
			assert X_train.shape[1] == self.tree.n_features_
		except:
			raise ValueError('The dim of X_train does not match n_features')
		
		try:
			assert X_train.shape[0] == len(y_train.flatten())
		except:
			raise ValueError('Length of X_train is not equal to y_train')
		
		# 逐样本统计decision_path.
		# 获得的decision_path_matrix为csr矩阵格式,需要转为np.ndarray, shape = (N, D).
		decision_path_matrix_ = self.tree.decision_path(X_train)
		decision_path_matrix_ = decision_path_matrix_.toarray()
		
		# decision_path按照 0-1 label分别统计.
		path_matrix_pos_ = decision_path_matrix_[list(np.argwhere(y_train == 1.0).flatten()), :]
		path_matrix_neg_ = decision_path_matrix_[list(np.argwhere(y_train == 0.0).flatten()), :]
		
		pos_feature_counts_ = path_matrix_pos_.sum(axis = 0)
		neg_feature_counts_ = path_matrix_neg_.sum(axis = 0)
		
		nodes_pos_neg_counts, nodes_pos_neg_ratio = {}, {}
		for i in range(self.node_count):
			pos_counts_, neg_counts_ = pos_feature_counts_[i], neg_feature_counts_[i]
			pos_ratio_ = pos_counts_ / (pos_counts_ + neg_counts_)
			nodes_pos_neg_counts.update({i: {'pos': pos_counts_, 'neg': neg_counts_}})
			nodes_pos_neg_ratio.update({i: {'pos': pos_ratio_, 'neg': 1.0 - pos_ratio_}})
		
		self.nodes_pos_neg_counts = nodes_pos_neg_counts
		self.nodes_pos_neg_ratio = nodes_pos_neg_ratio
	
	def cal_FC_for_sample(self, x_test: np.ndarray):
		"""
		计算预测某样本时的各特征贡献分数
		"""
		try:
			x_test = x_test.reshape(1, self.tree.n_features_)
		except:
			raise ValueError('The dim of x_test is not equal to n_features_')
			
		# 首先计算decision_path.
		decision_path_matrix_ = self.tree.decision_path(x_test)
		decision_path_matrix_ = decision_path_matrix_.toarray()
		
		# 获取decision_path上的节点id和相关特征id.
		decision_node_ids_ = list(np.argwhere(decision_path_matrix_ == 1.0)[:, 1])
		decision_features_ = [self.nodes_split_features[i] for i in decision_node_ids_]
		
		# 获取孪生children_node信息.
		twinborn_children_ids_ = []
		for i in range(len(decision_node_ids_) - 1):
			parent_id_ = decision_node_ids_[i]
			connects_ = self.nodes_connects[parent_id_]  # 获取i为parent_node时的连接
			children_ids_ = list(connects_.values())
			children_ids_.remove(decision_node_ids_[i + 1])
			twinborn_children_ids_.append(children_ids_[0])
		
		# 反向求解各特征分数.
		scores, features = {}, {}
		for i in range(-1, -len(decision_node_ids_), -1):
			# 获取parent和children的id.
			parent_id_ = decision_node_ids_[i - 1]
			deci_child_id_ = decision_node_ids_[i]
			twin_child_id_ = twinborn_children_ids_[i]
			
			# 记录feature.
			features.update({deci_child_id_: self.nodes_split_features[deci_child_id_]})
			if i == -len(decision_node_ids_) + 1:
				features.update({parent_id_: self.nodes_split_features[parent_id_]})
			
			# 记录children_score.
			for id_ in [deci_child_id_, twin_child_id_]:
				if id_ in scores.keys():
					pass
				else:
					scores.update({id_: self.nodes_pos_neg_ratio[id_]['pos']})
			
			# 记录parent_score.
			N1 = sum(self.nodes_pos_neg_counts[deci_child_id_].values())
			N2 = sum(self.nodes_pos_neg_counts[twin_child_id_].values())
			p_score_ = (scores[deci_child_id_] * N1 + scores[twin_child_id_] * N2) / (N1 + N2)
			scores.update({parent_id_: p_score_})
		
		# 计算此样本的FC值.
		FC = {}
		for i in range(len(decision_node_ids_) - 1):
			parent_id_ = decision_node_ids_[i]
			deci_feature_ = features[parent_id_]
			parent_score_ = scores[parent_id_]
			child_score_ = scores[decision_node_ids_[i + 1]]
			FC.update({deci_feature_: child_score_ - parent_score_})
			
		return FC
		

class LocalInterpForGBDT(object):
	"""
	GBDT的局部解释
	"""
	
	def __init__(self):
		pass
	
	def train_gbdt(self, X_train, y_train, params):
		self.X_train, self.y_train = X_train, y_train
		self.gbdt, self.tree_weights = _train_gbdt(X_train, y_train, params)
		
	def cal_FC_for_all_trees(self, x_test):
		FCs = {}
		for i in range(self.gbdt.n_estimators):
			tree_ = self.gbdt.estimators_[i, 0]
			dti = DecisionTreeInfo(tree_)
			dti.cal_nodes_pos_neg_counts_ratio(self.X_train, self.y_train)
			FCs.update({i: dti.cal_FC_for_sample(x_test)})
		return FCs
	

if __name__ == '__main__':
	# %% 载入数据和处理.
	from lib.tmp import X_train, y_train, x_test

	# %% 训练GBDT模型.
	params = {
		'n_estimators': 200,
		'learning_rate': 0.1,
		'min_samples_split': 10,
		'min_samples_leaf': 1,
		'max_depth': 4,
		'subsample': 1,
		'random_state': 10,
		'loss': 'deviance'
	}
	self = LocalInterpForGBDT()
	self.train_gbdt(X_train, y_train, params)
	FCs = self.cal_FC_for_all_trees(x_test)
	
	
	# y_train_pred = gbdt.predict(X_train)
	#
	# # 对比真实和预测值.
	# plt.figure()
	# plt.subplot(2, 1, 1)
	# plt.plot(y_train)
	# plt.subplot(2, 1, 2)
	# plt.plot(y_train_pred)
	#
	# # %% 获得GBDT中的tree信息.
	# gbdt_trees = {}
	# for i in range(gbdt.n_estimators):
	# 	gbdt_trees.update({i: gbdt.estimators_[i, 0]})
	#
	# # %% 决策树信息计算.
	# tree = gbdt_trees[0]
	# self = DecisionTreeInfo(tree)
	# self.cal_nodes_pos_neg_counts_ratio(X_train, y_train)
	
	



