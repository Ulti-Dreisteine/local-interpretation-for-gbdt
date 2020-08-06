# -*- coding: utf-8 -*-
"""
Created on 2020/3/25 11:26

@Project -> File: local-interpretation-for-gbdt -> __init__.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe:
"""

import logging

logging.basicConfig(level = logging.INFO)

import sys, os

sys.path.append('../')

from mod.config.config_loader import config_loader

proj_dir, proj_cmap = config_loader.proj_dir, config_loader.proj_cmap



