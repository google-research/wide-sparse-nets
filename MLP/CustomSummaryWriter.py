# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torch.utils.tensorboard import SummaryWriter as TSW
import torch

class CustomSummaryWriter:
  def __init__(self, savedir):
    self.localdir = f'runs/{savedir}'
    self.tsw = TSW(self.localdir)
    self.stat = dict()

  def add_data(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
    if main_tag not in self.stat:
      self.stat[main_tag]=dict()
    if global_step not in self.stat[main_tag]:
      self.stat[main_tag][global_step]=dict()
    for key, value in tag_scalar_dict.items():
      self.stat[main_tag][global_step][key] = value
    torch.save(self.stat, f'{self.localdir}/stats.pth')

  def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
      self.tsw.add_scalars(main_tag, tag_scalar_dict, global_step=global_step, walltime=walltime)
      self.add_data(main_tag, tag_scalar_dict, global_step=global_step, walltime=walltime)

  def close(self):
    self.tsw.close()

