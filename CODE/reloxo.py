# -*- coding: utf-8 -*-
"""
@author: pablo
"""

import time
from datetime import datetime

class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()
    def elapsed(self,sec):
        if sec < 60:
            return str(round(sec,2)) + " sec"
        elif sec < (60 * 60):
            return str(round(sec / 60,2)) + " min"
        else:
            return str(round(sec / (60 * 60),2)) + " hr"
    def elapsed_time(self):
        return self.elapsed(time.time() - self.start_time)
    def current_time(self):
        return datetime.fromtimestamp(time.time()).strftime("%d-%m-%Y %H:%M:%S")
    def reset(self):
        self.start_time = time.time()
