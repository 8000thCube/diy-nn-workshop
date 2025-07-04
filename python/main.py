from layers import Layer, Linear, Tanh
from typing import TypeAlias
import numpy as np
Value:TypeAlias = np.ndarray
class NN(Layer):
	"""basic neural network example struct"""
	def __init__(self,inputdimension:int,intermediatedimension:int,outputdimension:int):
		self.a=Tanh()
		self.l0=Linear(True,inputdimension,intermediatedimension)
		self.l1=Linear(False,intermediatedimension,outputdimension)
	def backward(self,cache:list[Value],ygrad:list[Value])->list[Value]:
		g=self.l1.backward(cache,ygrad);
		g=self.a.backward(cache,g);
		g=self.l0.backward(cache,g);
		return g
	def forward(self,cache:list[Value],x:list[Value])->list[Value]:
		x=self.l0.forward(cache,x)
		x=self.a.forward(cache,x)
		x=self.l1.forward(cache,x)
		return x
	def opt_step(self,f:callable([[Value,Value],tuple[Value,Value]])):
		self.a.opt_step(f)
		self.l0.opt_step(f)
		self.l1.opt_step(f)
def main():
	i=np.array([[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]]);
	lr=0.01
	persistence=0.5
	target=np.array([[0.0],[1.0],[1.0],[0.0]])
	nn=NN(2,10,1)

	for n in range(0,1000):
		cache=[]
		output=nn.forward(cache,[i])
		gradient=[(target-output[0])*2.0]
		nn.backward(cache,gradient)
		nn.momentum_sgd(lr,persistence)

	print(nn.infer([i])[0])
main()
