from collections import *
import random
Experience = namedtuple('Experience',('states','actions','next_states','rewards') )

class ReplayMemory:
	def __init__(self):
		self.capacity = 2048
		self.memory = []
		self.position = 0
		
	def push(self, *args ):
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		self.memory[self.position] = Experience(*args )
		self.position = (self.position + 1) % self.capacity
	
	def __len__(self):
		return len(self.memory )

	def sample(self, batch_size):
		return random.sample(self.memory, min(batch_size, len(self) ) )