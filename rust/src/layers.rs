impl Layer for Linear{
	fn backward(&mut self,cache:&mut Vec<Value>,outputgrad:Vec<Value>)->Vec<Value>{
		let range=outputgrad.len()..cache.len();
		cache.drain(range).zip(outputgrad).map(|(i,og)|{
			let db=og.clone();
			let di=og.clone().matmul(self.weight.clone().transpose());
			let dw=i.transpose().matmul(og);

			if let Some(weightgrad)=&mut self.weightgrad{*weightgrad+=dw}else{self.weightgrad=Some(dw)}
			if self.bias.is_some(){
				if let Some(biasgrad)=&mut self.biasgrad{*biasgrad+=db}else{self.biasgrad=Some(db)}
			}
			di
		}).collect()
	}
	fn forward(&self,cache:&mut Option<Vec<Value>>,input:Vec<Value>)->Vec<Value>{
		if let Some(c)=cache{c.extend(input.iter().cloned())}
		let output=input.into_iter().map(|i|{
			let mut o=i.matmul(self.weight.clone());
			if let Some(bias)=&self.bias{o+=bias}
			o
		}).collect();
		output
	}
	fn opt_step(&mut self,f:&mut dyn FnMut(Value,Value)->(Value,Value)){
		if let (weight,Some(weightgrad))=(&mut self.weight,&mut self.weightgrad){
			let (w,g)=f(take(weight),take(weightgrad));
			(self.weight,self.weightgrad)=(w,Some(g));
		}else{
			f(Value::default(),Value::default());
		}
		if let (Some(bias),Some(biasgrad))=(&mut self.bias,&mut self.biasgrad){
			let (b,g)=f(take(bias),take(biasgrad));
			(self.bias,self.biasgrad)=(Some(b),Some(g));
		}else{
			f(Value::default(),Value::default());
		}
	}
}
impl Layer for Tanh{
	fn backward(&mut self,cache:&mut Vec<Value>,outputgrad:Vec<Value>)->Vec<Value>{
		let range=outputgrad.len()..cache.len();
		cache.drain(range).zip(outputgrad).map(|(o,g)|o.map_2(|o,g|(1.0-o*o)*g,g)).collect()
	}
	fn forward(&self,cache:&mut Option<Vec<Value>>,input:Vec<Value>)->Vec<Value>{
		let output:Vec<Value>=input.into_iter().map(|i|i.map(f32::tanh)).collect();
		if let Some(c)=cache{c.extend(output.iter().cloned())}
		output
	}
	fn opt_step(&mut self,_f:&mut dyn FnMut(Value,Value)->(Value,Value)){}
}
impl Linear{
	pub fn new(bias:bool,inputdim:usize,outputdim:usize)->Self{
		let mut seed=tseed();
		let bias:Option<Vec<f32>>=bias.then(||(0..outputdim).map(|_|rfloat(&mut seed)).collect());
		let bias=bias.map(|b|Value::from_data(b,[outputdim]));
		let biasgrad=None;
		let weight:Vec<f32>=(0..inputdim*outputdim).map(|_|rfloat(&mut seed)).collect();
		let weight=Value::from_data(weight,[inputdim,outputdim]);
		let weightgrad=None;
		Self{bias,biasgrad,weight,weightgrad}
	}
}
impl Tanh{
	pub fn new()->Self{Tanh}
}
/// prng float on >=-1 and <1
pub fn rfloat(seed:&mut u128)->f32{
	update_seed(seed);
	*seed as i32 as f32*2.0_f32.powi(-31)
}
pub fn tseed()->u128{
	let mut seed=SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos();
	update_seed(&mut seed);
	update_seed(&mut seed);
	update_seed(&mut seed);
	seed
}/// prng seed updating
pub fn update_seed(seed:&mut u128){
	*seed^=*seed<<15;
	*seed^=*seed>>4;
	*seed^=*seed<<21;
}
#[derive(Clone,Debug)]
/// linear (matmul) layer
pub struct Linear{bias:Option<Value>,biasgrad:Option<Value>,weight:Value,weightgrad:Option<Value>}
#[derive(Clone,Debug)]
/// tanh layer
pub struct Tanh;
/// basic nn layer trait
pub trait Layer{
	/// applies the backward pass operation
	fn backward(&mut self,cache:&mut Vec<Value>,outputgrad:Vec<Value>)->Vec<Value>;
	/// applies the forward pass operation
	fn forward(&self,cache:&mut Option<Vec<Value>>,input:Vec<Value>)->Vec<Value>;
	/// applies the inference pass operation
	fn infer(&mut self,input:Vec<Value>)->Vec<Value>{self.forward(&mut None,input)}
	/// applys sgd optimization
	fn momentum_sgd(&mut self,lr:f32,persistence:f32){self.opt_step(&mut |param,paramgrad|(param.map_2(|p,g|g*lr+p,paramgrad.clone()),paramgrad*persistence))}
	/// adjusts the parameters according to the optimization function (param, gradient) -> new param, new gradient. this should call the function a consitent number of times between backward steps
	fn opt_step(&mut self,f:&mut dyn FnMut(Value,Value)->(Value,Value));
}
use crate::value::Value;
use std::{mem::take,time::SystemTime};
