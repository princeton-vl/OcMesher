# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma

from datetime import datetime
import os

import psutil


class Timer:

    def __init__(self, desc, disable_timer=False):
        self.disable_timer = disable_timer
        if self.disable_timer:    
            return
        self.name = f'[{desc}]'

    def __enter__(self):
        if self.disable_timer:
            return
        self.start = datetime.now()


    def __exit__(self, exc_type, exc_val, traceback):
        if self.disable_timer:
            return
        self.end = datetime.now()
        self.duration = self.end - self.start # timedelta
        if exc_type is None:
            process = psutil.Process(os.getpid())
            print(f'{self.name} finished in {str(self.duration)} with memory usage {process.memory_info().rss / 1024**3} GB')
        else:
            print(f'{self.name} failed with {exc_type}')


