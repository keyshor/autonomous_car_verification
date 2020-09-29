class Controller():
	def __init__(self, params):
		self.pee = params[0]
		self.eye = params[1]
		self.dee = params[2]
		self.thresh = params[3]
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
		rightView = observation[0][0:mid]
		leftView = observation[0][mid+1:]
		numRight = sum([int(a > self.thresh) for a in rightView])
		numLeft = sum([int(a > self.thresh) for a in leftView])
		err = numLeft - numRight
		return err

	def update_parameters(self, params):
		self.pee = params[0]
		self.eye = params[1]
		self.dee = params[2]
		self.thresh = params[3]
		self.prevErr = 0