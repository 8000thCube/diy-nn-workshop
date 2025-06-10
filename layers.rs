impl Layer for Tanh{
	fn backward(&mut self,input:&[Value],inputgrad:&mut [Value],outputgrad:&[Value]){
		todo!()
	}
	fn forward(&self,input:&[Value],output:&mut [Value]){input.iter().cloned().zip(output.iter_mut()).for_each(|(i,o)|*o=i.map(f32::tanh))}
}
pub struct Tanh;
/// basic nn layer trait
pub trait Layer{
	/// applies the backward pass operation
	fn backward(&mut self,input:&[Value],inputgrad:&mut [Value],outputgrad:&[Value]);
	/// applies the forward pass operation
	fn forward(&self,input:&[Value],output:&mut [Value]);
}
use crate::value::Value;
