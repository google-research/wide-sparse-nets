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

from torch.utils.tensorboard import SummaryWriter
import torch
import os
import time
import shutil
from datetime import datetime


def mk_alldir(dirpath):
	dir_list = dirpath.split('/')
	path = ''
	for d in dir_list:
		path = os.path.join(path, d)
		if not os.path.exists(path):
			os.mkdir(path)


class Recorder(SummaryWriter):
	"""
	A recorder for saving and loading experiment stats and checkpoints, 
	using TensorBoard's SummaryWriter.
	"""
	def __init__(self, out_dir=None, comment='', max_queue=10, flush_secs=120, save_ckpt_freq=10):
		"""
		Parameters
		  out_dir (str) - name of the folder for saving the output of the given experiment.
		  comment (str) - a suffix appended to the log_dir.
		  max_queue (int) - size of the queue for the light objects before saving them to file.
		  flush_secs (int) - how often the light objects will be saved in the stats file.
		  save_ckpt_freq (int) - frequency of saving checkpoints.
		"""

		self.out_dir = out_dir
		if self.out_dir is None:
			now = datetime.now()
			self.out_dir = f"{os.uname()[1].split('.')[0]}/"
			self.out_dir+= now.strftime("%Y_%m_%d_%H_%M_%S")
		if len(comment) > 0: self.out_dir+= f'-{comment}'
		
		print(f'Output will be saved to: {self.out_dir}', flush=True)


		self.fname_stats = 'stats.pt'
		self.run_local = f'runs/{self.out_dir}'
		self.ckpt_local = f'checkpoints/{self.out_dir}'
		self.max_queue = max_queue
		self.flush_secs = flush_secs
		self.queue_size = 0
		self.last_flush = time.time()
		self.save_ckpt_freq = save_ckpt_freq
		self.best_acc = 0
		self.stats= dict()

		mk_alldir(self.run_local)
		mk_alldir(self.ckpt_local)

		# calls SummaryWriter who saves TB event files to its log_dir
		super().__init__(log_dir=self.run_local, max_queue=max_queue, flush_secs=flush_secs)



	def add_to_stats(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
		"""
		Adds data to the stats file and saving (based on queue and flush status).
		"""
		if main_tag not in self.stats: 
			self.stats[main_tag]= dict()
		if global_step not in self.stats[main_tag]: 
			self.stats[main_tag][global_step]= dict()
		
		for key, value in tag_scalar_dict.items():
			self.stats[main_tag][global_step][key] = value
			self.queue_size+= 1

		if self.queue_size >= self.max_queue or time.time() - self.last_flush >= self.flush_secs:
			torch.save(self.stats, f'{self.run_local}/{self.fname_stats}')
			self.queue_size= 0
			self.last_flush= time.time()


	def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
		"""
		Writes data to TB files and the stats file.
		"""
		super().add_scalars(main_tag, tag_scalar_dict, global_step, walltime)
		self.add_to_stats(main_tag, tag_scalar_dict, global_step, walltime)



	def add_epoch_result(self, train_loss, test_loss, train_acc1, test_acc1,
						 train_acc5=None, test_acc5=None, 
						 global_step=None, walltime=None):
		"""
		Adds epoch evaluation results (losses and accuracies) to TB files and the stats file.
		"""
		self.add_scalars('loss', {'train': train_loss, 'test': test_loss}, 
						 global_step=global_step, walltime=walltime)
		self.add_scalars('acc1', {'train': train_acc1, 'test': test_acc1}, 
						 global_step=global_step, walltime=walltime)
		
		if train_acc5 or test_acc5:
			self.add_scalars('acc5', {'train': train_acc5, 'test': test_acc5}, 
						 	 global_step=global_step, walltime=walltime)



	def save_checkpoint(self, state_dict, global_step=None, is_best=False, prefix='model'):
		""" Subfunction to save_full_checkpoint(); saves model checkpoint. """
		if global_step==0: # init checkpoint
			mpath= f'{self.ckpt_local}/{prefix}_init.pt'
			torch.save(state_dict, mpath)

		elif global_step==-1: # final checkpoint
			mpath= f'{self.ckpt_local}/{prefix}_final.pt'
			torch.save(state_dict, mpath)
			if is_best: shutil.copyfile(mpath_last, mpath_best)

		elif global_step is None or global_step%self.save_ckpt_freq == 0:
			mpath_last= f'{self.ckpt_local}/{prefix}_last.pt'
			mpath_best= f'{self.ckpt_local}/{prefix}_best.pt'
			torch.save(state_dict, mpath_last)
			if is_best: shutil.copyfile(mpath_last, mpath_best)



	def save_full_checkpoint(self, model, optimizer, scheduler, args, epoch, test_acc):
		""" Saves model checkpoint. """
		is_best= test_acc > self.best_acc
		self.best_acc= max(test_acc, self.best_acc)
		self.save_checkpoint({'epoch': epoch,
							  'args': args,
							  'current_acc': test_acc,
							  'best_acc': self.best_acc,
							  'state_dict': model.state_dict(),
						      'optimizer' : optimizer.state_dict(),
							  'scheduler': scheduler.state_dict()}, 
							  global_step=epoch, is_best=is_best)



	def load_checkpoint(self, prefix='model', global_step=None, log_dir=None):

		#ckpt_dir= f'checkpoints/{self.out_dir}' if log_dir is None else f'checkpoints/{log_dir}'
		global_step= 'last' if global_step is None else global_step

		if global_step==0:
			dest= f'{ckpt_dir}/{prefix}_init.pt'
			src = f'gs://{self.bucket_dir}/{ckpt_dir}/{prefix}_init.pt'
		elif global_step=='best':
			dest= f'{ckpt_dir}/{prefix}_{global_step}.pt'
			src = f'gs://{self.bucket_dir}/{ckpt_dir}/{prefix}_{global_step}.pt'     
		else:
			dest= f'{ckpt_dir}/{prefix}_{global_step}.pt'
			src = f'gs://{self.bucket_dir}/{ckpt_dir}/{prefix}_last.pt'

		if not os.path.exists(dest):
			print(f'Path {dest} does not exist!\n>>> Attempting to download it from bucket ...')

		if os.path.exists(dest): # ?!?!?
				ckpt= torch.load(dest)
				print(f'--> Checkpoint loaded from {dest}!')
		else:
				print(f'--> No Checkpoint was found in {dest} or {src}!')
				ckpt= None
		return ckpt


	def resume_full_checkpoint(self, resume, model, optimizer, scheduler):
		if resume:
			if resume=='init':
				print(resume)
				full_dict= self.load_checkpoint(global_step=0)
			elif resume=='best':
				print(resume)
				full_dict= self.load_checkpoint(global_step='best')
			elif resume=='previous':
				full_dict= self.load_checkpoint()
			else:
				full_dict= self.load_checkpoint(log_dir=resume)

			model.load_state_dict(full_dict['state_dict'])
			optimizer.load_state_dict(full_dict['optimizer'])
			scheduler.load_state_dict(full_dict['scheduler'])
			self.best_acc= full_dict['best_acc']
			print(f'best_acc from model checkpoint is {self.best_acc} and has format: {type(self.best_acc)}')
			print(f'and epoch is = {full_dict["epoch"]}')
			return full_dict['epoch']
		return 0


	def close(self):
		super().flush()
		super().close()
		torch.save(self.stats, f'{self.run_local}/{self.fname_stats}')

