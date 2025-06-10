/// gets mutable data
fn buffer_mut<'a>(buffer:&'a mut Arc<[f32]>,count:usize,offset:&mut usize)->&'a mut [f32]{
	if Arc::strong_count(&*buffer)==1{return &mut Arc::get_mut(buffer).unwrap()[*offset..count+*offset]}
	let b=Arc::from(&buffer[*offset..count+*offset]);
	*buffer=b;
	*offset=0;
	Arc::get_mut(buffer).unwrap()
}
impl Value{
	/// creates from data and dimensions
	pub fn from_data<A:AsRef<[f32]>,B:AsRef<[usize]>>(data:A,dims:B)->Self{
		let (data,dims)=(data.as_ref(),dims.as_ref());
		let (buffer,dims):(Arc<[f32]>,Arc<[usize]>)=(Arc::from(data),Arc::from(dims));
		let mut strides=vec![0;dims.len()];
		let offset=0;

		let count=dims.iter().zip(strides.iter_mut()).rev().fold(1,|product,(&dim,stride)|{
			*stride=product;
			product*dim
		});
		let (buffer,dims,strides)=(Arc::from(data),Arc::from(dims),Arc::from(strides));
		Self{buffer,count,dims,offset,strides}
	}
	/// applies the function to each component
	pub fn map<F:Fn(f32)->f32>(mut self,f:F)->Value{
		buffer_mut(&mut self.buffer,self.count,&mut self.offset).iter_mut().for_each(|x|*x=f(*x));
		self
	}

}
#[derive(Clone,Debug,Default)]
/// nn layer io value. stores data and dimensions
pub struct Value{buffer:Arc<[f32]>,count:usize,dims:Arc<[usize]>,offset:usize,strides:Arc<[usize]>}
use std::sync::Arc;
