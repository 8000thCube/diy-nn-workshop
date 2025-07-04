fn main(){
	let input=Value::from_data([0.0,0.0, 0.0,1.0, 1.0,0.0, 1.0,1.0],[4,2]);
	let lr=0.01;
	let mut cache=Some(Vec::new());
	let persistence=0.5;
	let target=Value::from_data([0.0, 1.0, 1.0, 0.0],[4,1]);
	let mut nn=NN::new(2,10,1);

	for _ in 0..1000{
		let output=nn.forward(&mut cache,vec![input.clone()]);
		let gradient=vec![(&target-&output[0])*2.0];
		nn.backward(cache.as_mut().unwrap(),gradient);
		nn.momentum_sgd(lr,persistence);
	}

	nn.infer(vec![input]).into_iter().next().unwrap().into_data().into_iter().for_each(|output|println!("{output}"));
}
impl Layer for NN{
	fn backward(&mut self,cache:&mut Vec<Value>,outputgrad:Vec<Value>)->Vec<Value>{
		let g=self.l1.backward(cache,outputgrad);
		let g=self.a.backward(cache,g);
		let g=self.l0.backward(cache,g);
		g
	}
	fn forward(&self,cache:&mut Option<Vec<Value>>,input:Vec<Value>)->Vec<Value>{
		let x=self.l0.forward(cache,input);
		let x=self.a.forward(cache,x);
		let x=self.l1.forward(cache,x);
		x
	}
	fn opt_step(&mut self,f:&mut dyn FnMut(Value,Value)->(Value,Value)){
		self.a.opt_step(f);
		self.l0.opt_step(f);
		self.l1.opt_step(f);
	}
}
impl NN{
	fn new(inputdim:usize,intermediatedim:usize,outputdim:usize)->Self{
		Self{a:Tanh::new(),l0:Linear::new(true,inputdim,intermediatedim),l1:Linear::new(false,intermediatedim,outputdim)}
	}
}
#[derive(Clone,Debug)]
/// basic neural network example struct
pub struct NN{a:Tanh,l0:Linear,l1:Linear}
pub mod layers;
pub mod value;
use {
	layers::{Layer,Linear,Tanh},value::Value
};
