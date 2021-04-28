# -*- coding: utf-8 -*-
"""
Created on 2020/8/6 11:32 上午

@File: decision_tree_info.py

@Department: AI Lab, Rockontrol, Chengdu

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 决策树信息提取
"""

import logging

logging.basicConfig(level = logging.INFO)

from sklearn.tree import DecisionTreeRegressor
from collections import defaultdict
import numpy as np


def _get_decision_path_matrix(tree: DecisionTreeRegressor, X: np.ndarray) -> np.ndarray:
    """
    获取决策路径矩阵，路径上的节点对应值为1, 不在该路径上的节点对应值为0
    """
    try:
        X = X.reshape(-1, tree.n_features_)
    except:
        raise ValueError('X connot convert into shape = (-1, n_features).')
    
    decision_path_matrix = tree.decision_path(X)
    decision_path_matrix = decision_path_matrix.toarray()
    return decision_path_matrix


class DecisionTreeInfo(object):
    """决策树信息提取"""
    
    def __init__(self, tree: DecisionTreeRegressor):
        self.tree = tree
        self.node_count = tree.tree_.node_count  # 所有节点数目

        self.nodes_connects = self._get_nodes_connects()
        self.nodes_samples_number = self._get_nodes_samples_number()
        self.nodes_split_features = self._get_nodes_split_features()
        # self.nodes_split_thres = self._get_nodes_split_thres()  # 这个计算无用
    
    def _get_nodes_connects(self) -> dict:
        """
        获取父子节点连接关系nodes_connects, key为父节点编号, value分别对应左右分支子
        节点编号, -1代表没有子节点
        :return nodes_connects: dict, 如:
                        {
                                parent_id_0: {
                                        'left': left_child_id_0,
                                        'right': right_child_id_0
                                },
                                ...
                        }
        """
        children_left = self.tree.tree_.children_left
        children_right = self.tree.tree_.children_right
        nodes_connects = defaultdict(dict)
        for node_id in range(self.node_count):
            nodes_connects[node_id] = {
                'left': children_left[node_id],
                'right': children_right[node_id],
            }
        return nodes_connects

    def _get_nodes_samples_number(self) -> dict:
        """
        获取各节点上的样本总数
        :return nodes_samples_n: dict, 如:
                        {
                                node_id_0: N_0,
                                node_id_1: N_1,
                                ...
                        }
        """
        nodes_samples_n = defaultdict(int)
        for node_id in range(self.node_count):
            nodes_samples_n[node_id] = self.tree.tree_.n_node_samples[node_id]
        return nodes_samples_n
    
    def _get_nodes_split_features(self) -> dict:
        """
        获取各节点用于分裂的特征id, -2表示无分裂
        :return nodes_split_features: dict, 如:
                        {
                                node_id_0: feature_id_on_node_0,
                                node_id_1: feature_id_on_node_1,
                                ...
                        }
        """
        nodes_split_features = defaultdict(int)
        for node_id in range(self.node_count):
            nodes_split_features[node_id] = self.tree.tree_.feature[node_id]
        return nodes_split_features

    def _get_nodes_split_thres(self) -> dict:
        """
        获取各节点用于分裂的特征值记录
        :return nodes_split_thres: dict, 如:
                        {
                                node_id_0: thres_0,
                                node_id_1: thres_1,
                                ...
                        }
        """
        nodes_split_thres = defaultdict(float)
        for node_id in range(self.node_count):
            nodes_split_thres[node_id] = self.tree.tree_.threshold[node_id]
        return nodes_split_thres
    
    # def _get_nodes_connects(self) -> dict:
    # 	"""
    # 	获取父子节点连接关系nodes_connects, key为父节点编号, value分别对应左右分支子
    # 	节点编号, -1代表没有子节点
    # 	:return nodes_connects: dict, 如:
    # 			{
    # 				parent_id_0: {
    # 					'left': left_child_id_0,
    # 					'right': right_child_id_0
    # 				},
    # 				...
    # 			}
    # 	"""
    # 	nodes_connects = defaultdict(dict)
    # 	for node_id in range(self.node_count):
    # 		nodes_connects[node_id] = {
    # 			'left': self.tree.tree_.children_left[node_id],
    # 			'right': self.tree.tree_.children_right[node_id]
    # 		}
    # 	return nodes_connects
    
    # @property
    # def nodes_connects(self):
    # 	return self._get_nodes_connects()
    
    # def _get_nodes_samples_number(self) -> dict:
    # 	"""
    # 	获取各节点上的样本数量
    # 	:return nodes_samples_n: dict, 如:
    # 			{
    # 				node_id_0: N_0,
    # 				node_id_1: N_1,
    # 				...
    # 			}
    # 	"""
    # 	nodes_samples_n = defaultdict(int)
    # 	for node_id in range(self.node_count):
    # 		nodes_samples_n[node_id] = self.tree.tree_.n_node_samples[node_id]
    # 	return nodes_samples_n
    
    # @property
    # def nodes_samples_number(self):
    # 	return self._get_nodes_samples_number()
    
    # def _get_nodes_split_features(self) -> dict:
    # 	"""
    # 	获取各节点用于分裂的特征id, -2表示无分裂
    # 	:return nodes_split_features: dict, 如:
    # 			{
    # 				node_id_0: feature_id_on_node_0,
    # 				node_id_1: feature_id_on_node_1,
    # 				...
    # 			}
    # 	"""
    # 	nodes_split_features = defaultdict(int)
    # 	for node_id in range(self.node_count):
    # 		nodes_split_features[node_id] = self.tree.tree_.feature[node_id]
    # 	return nodes_split_features
    
    # @property
    # def nodes_split_features(self):
    # 	return self._get_nodes_split_features()
    
    # def _get_nodes_split_thres(self) -> dict:
    # 	"""
    # 	获取各节点用于分裂的特征值记录
    # 	:return nodes_split_thres: dict, 如:
    # 			{
    # 				node_id_0: thres_0,
    # 				node_id_1: thres_1,
    # 				...
    # 			}
    # 	"""
    # 	nodes_split_thres = defaultdict(float)
    # 	for node_id in range(self.node_count):
    # 		nodes_split_thres[node_id] = self.tree.tree_.threshold[node_id]
    # 	return nodes_split_thres
    
    # @property
    # def nodes_split_thres(self):
    # 	return self._get_nodes_split_thres()
    
    def _check_train_input(self, X_train, y_train):
        try:
            assert X_train.shape[1] == self.tree.n_features_
        except:
            raise ValueError('Dim of X_train does not match n_features.')
        
        try:
            assert X_train.shape[0] == len(y_train.flatten())
        except:
            raise ValueError('Length of X_train is not equal to y_train.')
    
    def cal_nodes_labels_counts_ratio(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        获取各节点上正负样本计数和占比
        :param X_train: 训练集特征X
        :param y_train: 训练集目标y

        Notes:
            目前这一步还需要X_train, y_train正向计算, 按理说当训练tree的时候就应该能
            获得节点上的正负样本信息, 但目前从sklearn代码中没有找到接口
        """
        self._check_train_input(X_train, y_train)
        _labels = np.unique(y_train)  # 所有的类别标签
        
        # 全量训练样本统计decision_path. 获得的decision_path_matrix为csr矩阵格式,需
        # 要转为np.ndarray, shape = (N = n_samples, D = n_features).
        _total_deci_path_matrix = _get_decision_path_matrix(self.tree, X_train)
        
        _total_labels_samples_counts = {}
        for _label in _labels:
            _deci_path_matrix = _total_deci_path_matrix[
                                np.argwhere(y_train == _label).flatten(), :
                                ]
            _total_labels_samples_counts[_label] = _deci_path_matrix.sum(axis = 0)
        
        nodes_labels_counts, nodes_labels_ratio = {}, {}
        for node_id in range(self.node_count):
            _node_labels_count, _node_labels_ratio = {}, {}
            
            # 首先统计该node各个label上的数目.
            for _label in _labels:
                _node_labels_count[_label] = _total_labels_samples_counts[_label][node_id]
            
            _count_sum = sum(_node_labels_count.values())
            for _label in _labels:
                _node_labels_ratio[_label] = _node_labels_count[_label] / _count_sum
                
            nodes_labels_counts[node_id] = _node_labels_count
            nodes_labels_ratio[node_id] = _node_labels_ratio
        
        return nodes_labels_counts, nodes_labels_ratio

    # def cal_nodes_labels_counts_ratio(self):
    #     """
    #     获取各节点上正负样本计数和占比
    #     :param X_train: 训练集特征X
    #     :param y_train: 训练集目标y
    #     """
    #     nodes_label_counts = self.tree.tree_.value
    #     nodes_labels_counts, nodes_labels_ratio = {}, {}
    #     for node_id in range(self.node_count):
    #         nodes_labels_counts[node_id] = {}
    #         label_counts_sum = 0
    #         for i in range(self.tree.n_classes_):  # TODO: 这里的输出是array而且还不对, 
    #             nodes_labels_counts[node_id][i] = int(nodes_label_counts[node_id][0][i])
    #             label_counts_sum += nodes_labels_counts[node_id][i]
            
    #         nodes_labels_ratio[node_id] = {}
    #         for i in range(self.tree.n_classes_):
    #             nodes_labels_ratio[node_id][i] = nodes_labels_counts[node_id][i] / label_counts_sum
    #     return nodes_labels_counts, nodes_labels_ratio
    
    def _check_test_input(self, x_test):
        try:
            x_test = x_test.reshape(1, self.tree.n_features_)
            return x_test
        except:
            raise ValueError('Dim of x_test is not equal to n_features_.')
    
    def cal_FC_for_sample(self, x_test: np.ndarray, nodes_labels_counts: dict,
                          nodes_labels_ratio: dict):
        """计算预测某样本时的各特征贡献分数"""
        x_test = self._check_test_input(x_test)
        
        # ---- 计算decision_path并获取path上的节点id和相关特征id -------------------------------------

        _deci_path_matrix = _get_decision_path_matrix(self.tree, x_test)
        _deci_node_ids = np.argwhere(_deci_path_matrix == 1.0)[:, 1].flatten()
        
        # ---- 获取_deci_node_ids中处于同一分岔的孪生节点id信息 --------------------------------------
        
        # _deci_node_ids的第一个节点一定是总parent节点,之后各节点一定会对应一个
        # 孪生分岔叶子节点id, 对应顺序地记录于_twinborn_children_ids中.
        _twinborn_children_ids = []
        for i in range(len(_deci_node_ids) - 1):
            _parent_id = _deci_node_ids[i]
            _connects = self.nodes_connects[_parent_id]  # 获取i为parent_node时的连接
            _children_ids = list(_connects.values())
            _children_ids.remove(_deci_node_ids[i + 1])  # 因为决策数每次分叉都是二分, 所以一定是一对id
            _twinborn_children_ids.append(_children_ids[0])
        
        # ---- 反向求解各特征分数 -------------------------------------------------------------------
        
        # 从分岔树底层逐层向上计算各节点分数.
        features, scores = {}, {}
        for i in range(-1, -len(_deci_node_ids), -1):
            # 获取parent和children的id.
            _parent_id = _deci_node_ids[i - 1]  # 上一级parent节点id
            _deci_child_id = _deci_node_ids[i]  # 同一级child节点id
            _twin_child_id = _twinborn_children_ids[i]  # 同一级孪生id
            
            # 记录各处于deci_path的节点上进行分岔的feature.
            features[_deci_child_id] = self.nodes_split_features[_deci_child_id]
            if i == -len(_deci_node_ids) + 1:
                features[_parent_id] = self.nodes_split_features[_parent_id]
                
            # 记录children_score. FC值来源于对特征LI值的计算, 而LI值由对应特征在
            # child和上一个parent节点上正样本比例差Y_{mean}^{c} - Y_{mean}^{p}计算
            # 获得, 参见原文第七页式(1)～式(3)内容. 原文强调GBDT的特征重要性计算leaf
            # score的差异主要在于根据样本不平衡性进行的分数加权处理.
            for node_id in [_deci_child_id, _twin_child_id]:
                if node_id in scores.keys():
                    pass
                else:
                    # ------------ TODO: 这里需要查阅GBDT如何计算node score的.
                    # <<<<<<<<<<<< 旧代码
                    scores[node_id] = nodes_labels_ratio[node_id][1]  # 使用子节点正样本比例做为分数
                    # >>>>>>>>>>>> from the pull request
                    # scores[node_id] = self.tree.tree_.value[node_id, 0, 0]  # 使用子节点的value做为分数
                    # ------------
                    
            # 记录parent_score.
            N1 = sum(nodes_labels_counts[_deci_child_id].values())
            N2 = sum(nodes_labels_counts[_twin_child_id].values())
            _parent_score = (scores[_deci_child_id] * N1 + scores[_twin_child_id] * N2) / (N1 + N2)
            scores[_parent_id] = _parent_score
            
        # 计算此样本的FC值.
        FC = {}
        for i in range(len(_deci_node_ids) - 1):
            _parent_id = _deci_node_ids[i]
            _deci_feature = features[_parent_id]
            _parent_score = scores[_parent_id]
            _child_score = scores[_deci_node_ids[i + 1]]
            FC[_deci_feature] = _child_score - _parent_score
        
        return FC
    

if __name__ == '__main__':
	from sklearn.ensemble import GradientBoostingClassifier
	import sys
	import os
    
	BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), '../../'))
	sys.path.append(BASE_DIR)
    
	from test_script.local_test import X_train, y_train, x_test
    
	# ---- 训练模型 ---------------------------------------------------------------------------------
    
	params = {
		'n_estimators': 50,
		'learning_rate': 0.1,
		'min_samples_split': 10,
		'min_samples_leaf': 2,
		'max_depth': 3,
		'subsample': 1,
		'random_state': 10,
		'loss': 'deviance',
	}
	clf = GradientBoostingClassifier(**params)
	clf.fit(X_train, y_train)
    
	# ---- 单决策树模型计算 -------------------------------------------------------------------------
    
	i = 0
	tree = clf.estimators_[i, 0]  # type: DecisionTreeRegressor
    
	self = DecisionTreeInfo(tree)
	nodes_labels_counts, nodes_labels_ratio = self.cal_nodes_labels_counts_ratio(X_train, y_train)
	FC = self.cal_FC_for_sample(x_test, nodes_labels_counts, nodes_labels_ratio)



