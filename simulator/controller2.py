import numpy as np

class Controller():
	def __init__(self, params):
		self.pee = params[0]
		self.eye = params[1]
		self.dee = params[2]
		self.prevErr = 0

	def predict(self, observation):
		err = self.errorFunc(observation)
		result = self.pee*err + self.dee*(err - self.prevErr)
		self.prevErr = err
		return result

	def reset(self):
		self.prevErr = 0

	def errorFunc(self, observation):
		mid = len(observation[0])//2
		leftView = observation[0][0:mid]
		rightView = observation[0][mid+1:]
		err = sum(rightView) - sum(leftView)
		return err

	def update_parameters(self, params):
		self.pee = params[0]
		self.eye = params[1]
		self.dee = params[2]
		self.thresh = params[3]
		self.prevErr = 0