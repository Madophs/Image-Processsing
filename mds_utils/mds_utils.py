#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 11:36:45 2021

@author: ruizvilj
"""

from sys import platform

class MdsUtils:
    def __construct__(self):
        self.system = "Unknown"
    
    def getPlatform(self) -> str:
        return platform