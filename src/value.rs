/// gets mutable data
fn buffer_mut<'a,E:Clone>(buffer:&'a mut Arc<[E]>,count:usize,offset:&mut usize)->&'a mut [E]{
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
	/// applies the function to each pair of components
	pub fn map_2<F:Fn(f32,f32)->f32>(mut self,f:F,r:Value)->Value{
		todo!()
	}
	/// applies broadcasting
	pub fn try_broadcast(mut self,dims:&[usize])->Option<Value>{
		if dims.iter().zip(self.dims.iter()).any(|(b,d)|b!=d&&d!=&1){return None}
		let (dl,sl)=(self.dims.len(),self.strides.len());
		let (d,s)=(buffer_mut(&mut self.dims,dl,&mut 0),buffer_mut(&mut self.strides,sl,&mut 0));

		d.iter_mut().zip(dims.iter()).zip(s.iter_mut()).for_each(|((d,b),s)|if d!=b{(*d,*s)=(*b,0)});
		Some(self)
	}
}

#[derive(Clone,Debug,Default)]
/// nn layer io value. stores data and dimensions
pub struct Value{buffer:Arc<[f32]>,count:usize,dims:Arc<[usize]>,offset:usize,strides:Arc<[usize]>}
use std::sync::Arc;
