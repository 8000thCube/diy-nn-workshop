from collections.abc import Callable
from typing import TypeAlias
import numpy as np
Value:TypeAlias = np.ndarray
class Layer:
	def backward(self,cache:list[Value],ygrad:list[Value])->list[Value]:
		"""applies the backward pass operation"""
		raise NotImplementedError
	def forward(self,cache:list[Value],x:list[Value])->list[Value]:
		"""applies the forward pass operation"""
		raise NotImplementedError
	def infer(self,x:list[Value])->list[Value]:
		return self.forward(None,x)
	def momentum_sgd(self,lr:float,persistence:float):
		"""
		Simple SGD with momentum
		"""
		self.opt_step(lambda param,grad:tuple([grad*lr+param,grad*persistence]))
	def opt_step(self,f:callable([[Value,Value],tuple[Value,Value]])):
		"""applies the optimization step to each parameter according to f(parameter, gradient) -> (newparameter, newgradient) """
		raise NotImplementedError
class Linear(Layer):
	def __init__(self,bias:bool,inputdimension:int,outputdimension:int):
		self.weight=np.random.randn(inputdimension,outputdimension)
		self.weightgrad=None
		self.bias=np.random.randn(outputdimension) if bias else None
		self.biasgrad=None
	def backward(self,cache:list[Value],ygrad:list[Value])->list[Value]:
		x=cache[len(cache)-len(ygrad):]
		xgrad=[]
		for x,ygrad in zip(x,ygrad):
			db=ygrad
			dw=np.matmul(x.transpose(),ygrad)
			dx=np.matmul(ygrad,self.weight.transpose())
			if self.weightgrad is None:
				self.weightgrad=dw
			else:
				self.weightgrad=self.weightgrad+dw
			if self.bias is not None:
				if self.biasgrad is None:
					self.biasgrad=db
				else:
					self.biasgrad=self.biasgrad+db
			xgrad.append(dx)
		del cache[len(cache)-len(ygrad):]
		return xgrad
	def forward(self,cache:list[Value],x:list[Value])->list[Value]:
		if cache is not None:
			cache.extend(x)
		x=list(map(lambda x:np.matmul(x,self.weight),x))
		if self.bias is not None:
			x=list(map(lambda x:x+self.bias,x))
		return x
	def opt_step(self,f:callable([[Value,Value],tuple[Value,Value]])):
		if self.weightgrad is not None:
			if len(self.weightgrad.shape)>len(self.weight.shape):
				for n in range(0,len(self.weightgrad.shape)-len(self.weight.shape)):
					self.weightgrad=np.mean(self.weightgrad,axis=0,keepdims=False)
			self.weight,self.weightgrad=f(self.weight,self.weightgrad)
		if self.bias is not None:
			if self.biasgrad is not None:
				if len(self.biasgrad.shape)>len(self.bias.shape):
					for n in range(0,len(self.biasgrad.shape)-len(self.bias.shape)):
						self.biasgrad=np.mean(self.biasgrad,axis=0,keepdims=False)
				self.bias,self.biasgrad=f(self.bias,self.biasgrad)
class Tanh(Layer):
	def backward(self,cache:list[Value],ygrad:list[Value])->list[Value]:
		y=cache[len(cache)-len(ygrad):]
		xgrad=[]
		for y,ygrad in zip(y,ygrad):
			xgrad.append(1.0-y*y)
		del cache[len(cache)-len(ygrad):]
		return xgrad
	def forward(self,cache:list[Value],x:list[Value])->list[Value]:
		y=list(map(lambda x:np.tanh(x),x))
		if cache is not None:
			cache=cache.extend(y)
		return y
	def opt_step(self,f:callable([[Value,Value],tuple[Value,Value]])):
		pass
