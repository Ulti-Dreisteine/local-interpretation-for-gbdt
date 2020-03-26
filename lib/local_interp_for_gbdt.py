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
import pandas as pd
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


def _get_decision_path_matrix(tree: DecisionTreeRegressor, X: np.ndarray) -> np.ndarray:
	"""
	获取决策路径矩阵，路径上的节点对应值为1, 不在该路径上的节点对应值为0
	"""
	try:
		X = X.reshape(-1, tree.n_features_)
	except:
		raise ValueError('X dim does not match n_features_')
		
	decision_path_matrix = tree.decision_path(X)
	decision_path_matrix = decision_path_matrix.toarray()
	return decision_path_matrix
	

class DecisionTreeInfo(object):
	"""
	决策树信息计算
	"""
	
	def __init__(self, tree: DecisionTreeRegressor):
		self.tree = tree
		self.node_count = tree.tree_.node_count
	
	def _get_nodes_connects(self) -> dict:
		"""
		获取父子节点连接关系nodes_connects, key为父节点编号, value分别对应左右分支子节点编号, -1代表没有子节点
		:return nodes_connects: dict,
			like {
					parent_id_0: {'left': left_child_id_0, 'right': right_child_id_0},
					...
				}
		"""
		nodes_connects = {}
		for node_id in range(self.node_count):
			nodes_connects.update(
				{
					node_id: {
						'left': self.tree.tree_.children_left[node_id],
						'right': self.tree.tree_.children_right[node_id]
					}
				}
			)
		return nodes_connects
	
	@property
	def nodes_connects(self):
		return self._get_nodes_connects()
	
	def _count_nodes_samples_number(self) -> dict:
		"""
		获取各节点上的样本数量
		:return nodes_samples_n: dict,
			like {
					node_id_0: N_0,
					node_id_1: N_1,
					...
				 }
		"""
		nodes_samples_n = {}
		for node_id in range(self.node_count):
			nodes_samples_n.update({node_id: self.tree.tree_.n_node_samples[node_id]})
		return nodes_samples_n
	
	@property
	def nodes_samples_number(self):
		return self._count_nodes_samples_number()
	
	def _get_nodes_split_thres(self) -> dict:
		"""
		获取各节点用于分裂的特征值记录
		:return nodes_split_thres: dict
			like {
					node_id_0: thres_0,
					node_id_1: thres_1,
					...
				 }
		"""
		nodes_split_thres = {}
		for node_id in range(self.node_count):
			nodes_split_thres.update({node_id: self.tree.tree_.threshold[node_id]})
		return nodes_split_thres
	
	@property
	def nodes_split_thres(self):
		return self._get_nodes_split_thres()
	
	def _get_nodes_split_features(self) -> dict:
		"""
		获取各节点用于分裂的特征id
		:return nodes_split_features: dict,
			like {
					node_id_0: feature_id_on_node_0,
					node_id_1: feature_id_on_node_1,
					...
				 }
		"""
		nodes_split_features = {}
		for node_id in range(self.node_count):
			nodes_split_features.update({node_id: self.tree.tree_.feature[node_id]})
		return nodes_split_features
	
	@property
	def nodes_split_features(self):
		return self._get_nodes_split_features()
	
	def cal_nodes_pos_neg_counts_ratio(self, X_train: np.ndarray, y_train: np.ndarray):
		"""
		获取各节点上正负样本计数和占比
		
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
		decision_path_matrix_ = _get_decision_path_matrix(self.tree, X_train)
		
		# decision_path按照 0-1 label分别统计.
		path_matrix_pos_ = decision_path_matrix_[np.argwhere(y_train == 1.0).flatten(), :]
		path_matrix_neg_ = decision_path_matrix_[np.argwhere(y_train == 0.0).flatten(), :]
		
		pos_samples_counts_ = path_matrix_pos_.sum(axis = 0)
		neg_samples_counts_ = path_matrix_neg_.sum(axis = 0)
		
		nodes_pos_neg_counts, nodes_pos_neg_ratio = {}, {}
		for node_id in range(self.node_count):
			pos_counts_, neg_counts_ = pos_samples_counts_[node_id], neg_samples_counts_[node_id]
			pos_ratio_ = pos_counts_ / (pos_counts_ + neg_counts_)
			nodes_pos_neg_counts.update({node_id: {'pos': pos_counts_, 'neg': neg_counts_}})
			nodes_pos_neg_ratio.update({node_id: {'pos': pos_ratio_, 'neg': 1.0 - pos_ratio_}})
		
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
		decision_path_matrix_ = _get_decision_path_matrix(self.tree, x_test)
		
		# 获取decision_path上的节点id和相关特征id.
		decision_node_ids_ = np.argwhere(decision_path_matrix_ == 1.0)[:, 1].flatten()
		# decision_features_ = [self.nodes_split_features[i] for i in decision_node_ids_]
		
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
			for node_id in [deci_child_id_, twin_child_id_]:
				if node_id in scores.keys():
					pass
				else:
					scores.update({node_id: self.nodes_pos_neg_ratio[node_id]['pos']})
			
			# 记录parent_score.
			N1 = sum(self.nodes_pos_neg_counts[deci_child_id_].values())
			N2 = sum(self.nodes_pos_neg_counts[twin_child_id_].values())
			parent_score_ = (scores[deci_child_id_] * N1 + scores[twin_child_id_] * N2) / (N1 + N2)
			scores.update({parent_id_: parent_score_})
		
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
		
	def _cal_FC_for_all_trees(self, x_test) -> dict:
		FC_results = {}
		for i in range(self.gbdt.n_estimators):
			tree_ = self.gbdt.estimators_[i, 0]
			dti = DecisionTreeInfo(tree_)
			dti.cal_nodes_pos_neg_counts_ratio(self.X_train, self.y_train)
			FC_results.update({i: dti.cal_FC_for_sample(x_test)})
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
	# %% 载入数据和处理.
	from lib.tmp import X_train, y_train, features, x_test

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
	local_interp_gbdt = LocalInterpForGBDT()
	local_interp_gbdt.train_gbdt(X_train, y_train, params)

	# %% 测试DecisionTreeInfo类.
	# tree = local_interp_gbdt.gbdt.estimators_[0, 0]
	# self = DecisionTreeInfo(tree)
	
	# %% 计算local interp score.
	top_features, top_FC_values = local_interp_gbdt.local_interpretation(x_test, features, top_n = 30)

	plt.figure()
	plt.bar(top_features, top_FC_values)
	plt.xticks(range(len(top_features)), top_features, rotation = 90, fontsize = 6)
	plt.yticks(fontsize = 6)
	plt.tight_layout()
	



