import random
"""------------------------ Heap Data Structure ------------------------ 
# This data structure is used to easily find the minimum/maximum if
# if you find yourself constantly calculating the minimum of a list.
# The last class generates and maintains two heaps (one min and one max)
# allowing for the median to quickly be found at any given moment"""
class MinHeap:
	def __init__(self, content = None):
		self.heap = []

	def __repr__(self):
		return f"Heap: {self.heap}"

	def insert(self, value):
		self.heap.append(value)
		value_position = len(self.heap)
		if len(self.heap) == 1: return
		while self.heap[(value_position >> 1) - 1] > value and value_position > 1:# Bubble-up while parent is larger
			self.heap[value_position - 1], self.heap[(value_position >> 1) - 1] = self.heap[(value_position >> 1) - 1], self.heap[value_position - 1]# Swap child and parent if necessary
			value_position = value_position >> 1 # Correct the value position

	def batch_insert(self, values):
		for value in values:
			self.insert(value)

	def extract(self): # Function probably has high constant and could be simplified
		if len(self.heap) == 0: return None
		
		self.heap[0], self.heap[-1] = self.heap[-1], self.heap[0]# Switch the first and last positions - Preserves structure
		min = self.heap.pop() # Pop mininimum value now at the end of the heap
		
		if len(self.heap) == 0: return min # If the heap is empty, return minimum
		
		value = self.heap[0] # Set value violating heap property as the first value
		position = 1 # Set position of value violating heap property
		while (position << 1) <= len(self.heap): # While value violating heap properity has children
			first_child = self.heap[(position << 1) - 1] # Grab the first child
			
			if (position << 1) < len(self.heap): second_child = self.heap[(position << 1)] # If there is a second child grab it
			
			else: second_child = float('inf') # Else declare the second child as infinity
			#print(f"{position} --> {position << 1} - {first_child} brother of {second_child} - {value} value")
			if first_child >= value and second_child >= value: break # If the heap property has been restored break loop
			
			if first_child <= second_child and first_child < value: # If first child is bigger than second child and larger than the value of interest
				self.heap[(position << 1) - 1], self.heap[position - 1] = self.heap[position - 1], self.heap[(position << 1) - 1] # Swap value of interest and first child
				position = (position << 1) # Srt new position of value of interest
			
			if first_child > second_child and second_child < value: # If second child is larger and the value is larger than the value of interest
				self.heap[(position << 1)], self.heap[position - 1] = self.heap[position - 1], self.heap[(position << 1)] # Swap value of interest and second child
				position = (position << 1) + 1 # Set new position of value of interest
		return min

	def find(self):
		return self.heap[0]

	def size(self):
		return len(self.heap)

class MaxHeap:
	def __init__(self, content = None):
		self.heap = []

	def __repr__(self):
		return f"Heap: {self.heap}"

	def insert(self, value):
		self.heap.append(value)
		value_position = len(self.heap)
		if len(self.heap) == 1: return # If the heap only has a single value in it
		while self.heap[(value_position >> 1) - 1] < value and value_position > 1:# Bubble-up while parent is larger
			self.heap[value_position - 1], self.heap[(value_position >> 1) - 1] = self.heap[(value_position >> 1) - 1], self.heap[value_position - 1]# Swap child and parent if necessary
			value_position = value_position >> 1 # Correct the value position

	def batch_insert(self, values):
		for value in values:
			self.insert(value)

	def extract(self): # Function has high constant and could be simplified
		if len(self.heap) == 0: return None
		
		self.heap[0], self.heap[-1] = self.heap[-1], self.heap[0]# Switch the first and last positions - Preserves structure
		
		max = self.heap.pop() # Pop mininimum value now at the end of the heap
		
		if len(self.heap) == 0: return max # If the heap is empty, return minimum
		
		value = self.heap[0] # Set value violating heap property as the first value
		position = 1 # Set position of value violating heap property
		while (position << 1) <= len(self.heap): # While value violating heap properity has children
			first_child = self.heap[(position << 1) - 1] # Grab the first child
			
			if (position << 1) < len(self.heap): second_child = self.heap[(position << 1)] # If there is a second child grab it
			
			else: second_child = float('-inf') # Else declare the second child as negative infinity
			
			if first_child <= value and second_child <= value: break # If the heap property has been restored break loop
			#print(f"First Child {first_child} - Second Child {second_child} - Value {value}")
			
			if first_child >= second_child and first_child > value: # If first child is bigger than second child and larger than the value of interest
				self.heap[(position << 1) - 1], self.heap[position - 1] = self.heap[position - 1], self.heap[(position << 1) - 1] # Swap value of interest and first child
				position = (position << 1) # Srt new position of value of interest
			
			if first_child < second_child and second_child > value: # If second child is larger and the value is larger than the value of interest
				self.heap[(position << 1)], self.heap[position - 1] = self.heap[position - 1], self.heap[(position << 1)] # Swap value of interest and second child
				position = (position << 1) + 1 # Set new position of value of interest
		return max

	def find(self):
		return self.heap[0]

	def size(self):
		return len(self.heap)

class MedianMaintainer:
	def __init__(self, content = None):
		self.UH = MinHeap(content = []) # Initialize an empty min heap as the upper heap
		self.LH = MaxHeap(content = []) # Initialize an empty max heap as the lower heap
		self.median = 0
		self.median_sum = 0
		self.batch_insert(content)

	def __repr__(self):
		return f"Lower Heap: {self.LH.heap}\nUpper Heap: {self.UH.heap}"


	def insert(self, value):
		if value < self.median:
			self.LH.insert(value)
		else:
			self.UH.insert(value)
		if abs(self.LH.size() - self.UH.size()) > 1:
			self.rebalance()
		self.median = self.find_median()
		self.median_sum += self.median
		return

	def batch_insert(self, contents):
		for value in contents:
			self.insert(value)
			self.__repr__

	def rebalance(self):
		if self.LH.size() > self.UH.size():
			self.UH.insert(self.LH.extract_max())
		else:
			self.LH.insert(self.UH.extract_min())

	def find_median(self):
		if self.UH.size() > self.LH.size(): # If the upper heap is bigger or equal to the lower heap size
			return self.UH.find_min() # Grab min value in upper heap
		else: # If the lower heap is bigger in size of the upper heap
			return self.LH.find_max() # Grab max value in lower heap
