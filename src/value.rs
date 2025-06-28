/// gets data from the arc buffer
fn buffer<'a,E>(buffer:&'a Arc<[E]>,count:usize,offset:&usize)->&'a [E]{&buffer[*offset..count+*offset]}
/// gets mutable data from the arc buffer
fn buffer_mut<'a,E:Clone>(buffer:&'a mut Arc<[E]>,count:usize,offset:&mut usize)->&'a mut [E]{
	if Arc::strong_count(&*buffer)==1{return &mut Arc::get_mut(buffer).unwrap()[*offset..count+*offset]}
	let b=Arc::from(&buffer[*offset..count+*offset]);
	*buffer=b;
	*offset=0;
	Arc::get_mut(buffer).unwrap()
}
impl AsRef<[usize]> for Shape{
	fn as_ref(&self)->&[usize]{&self.dims}
}
impl Deref for Shape{
	fn deref(&self)->&Self::Target{&self.dims}
	type Target=[usize];
}
impl Iterator for Coordinates{
	fn next(&mut self)->Option<Self::Item>{
		let dims=&self.dims;
		let positions=buffer_mut(&mut self.positions,dims.len(),&mut 0);

		if dims.iter().zip(positions.iter_mut()).all(|(d,x)|{
			let nextline=d==x;
			if nextline{*x=0}else{*x+=1}
			nextline
		}){
			None
		}else{
			Some(self.positions.clone())
		}
	}
	type Item=Arc<[usize]>;
}
impl Shape{
	/// applies broadcasting
	pub fn broadcast(mut self,dims:&[usize])->Shape{
		let mut error=false;
		let l=dims.len();
		let n=l.saturating_sub(self.len());
		self=self.unsqueeze(0,n);
		let (d,s)=(buffer_mut(&mut self.dims,l,&mut 0),buffer_mut(&mut self.strides,l,&mut 0));

		d.iter_mut().zip(dims.iter()).zip(s.iter_mut()).filter(|((d,b),_s)|d!=b).for_each(|((d,b),s)|if *d==1{(*d,*s)=(*b,0)}else if *b!=1{error=true});
		if error{panic!("{d:?} cannot be broadcast to {dims:?}")}
		self
	}
	/// adds n 1 dimensions at postion d
	pub fn unsqueeze(self,d:usize,n:usize)->Shape{
		if n==0{return self}
		let (dims,strides)=(&self.dims,&self.strides);
		let dims=dims[..d].iter().copied().chain((0..n).map(|_|1)).chain(dims[d..].iter().copied()).collect();
		let strides=strides[..d].iter().copied().chain((0..n).map(|_|0)).chain(strides[d..].iter().copied()).collect();
		Shape{dims,strides}
	}
}
impl Value{
	/// applies broadcasting
	pub fn broadcast(mut self,dims:&[usize])->Value{
		self.shape=self.shape.broadcast(dims);
		self
	}
	/// references the dimensions
	pub fn dims(&self)->&[usize]{&self.shape.dims}
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
		let shape=Shape{dims,strides};
		Self{buffer,count,offset,shape}
	}
	/// applies the function to each component
	pub fn map<F:Fn(f32)->f32>(mut self,f:F)->Value{
		buffer_mut(&mut self.buffer,self.count,&mut self.offset).iter_mut().for_each(|x|*x=f(*x));
		self
	}
	/// applies the function to each pair of components
	pub fn map_2<F:Fn(f32,f32)->f32>(self,f:F,r:Value)->Value{
		let mut l=self.broadcast(r.dims());
		let mut r=r.broadcast(l.dims());
		let (lbcnt,rbcnt)=((Arc::strong_count(&l.buffer)>1 as usize,l.count),(Arc::strong_count(&r.buffer)>1 as usize,r.count));

		if l.shape.strides==r.shape.strides{
			if lbcnt<rbcnt{
				buffer_mut(&mut l.buffer,l.count,&mut l.offset).iter_mut().zip(r.buffer[r.offset..r.count+r.offset].iter()).for_each(|(l,r)|*l=f(*l,*r));
				Value{buffer:l.buffer,count:l.count,offset:l.offset,shape:l.shape}
			}else{
				l.buffer[l.offset..l.count+l.offset].iter().zip(buffer_mut(&mut r.buffer,r.count,&mut r.offset).iter_mut()).for_each(|(l,r)|*r=f(*l,*r));
				Value{buffer:r.buffer,count:r.count,offset:r.offset,shape:r.shape}
			}
		}else{
			buffer_mut(&mut r.shape.dims,r.shape.strides.len(),&mut 0).fill(0);
			let lb=buffer_mut(&mut l.buffer,l.count,&mut l.offset);
			let rb=&r.buffer[r.offset..r.offset+r.count];
			for c in (Coordinates{
				dims:l.shape.dims.clone(),positions:r.shape.dims
			}){
				let lx:usize=c.iter().rev().zip(l.shape.strides.iter().rev()).map(|(c,l)|c*l).sum();
				let rx:usize=c.iter().rev().zip(r.shape.strides.iter().rev()).map(|(c,r)|c*r).sum();
				lb[lx]=f(lb[lx],rb[rx]);
			}
			Value{buffer:l.buffer,count:l.count,offset:l.offset,shape:l.shape}
		}
	}
	/// returns the tensor shape, which contains the dimensions
	pub fn shape(&self)->Shape{self.shape.clone()}
}
#[derive(Clone,Debug,Default)]
/// iterator over tensor positions
pub struct Coordinates{dims:Arc<[usize]>,positions:Arc<[usize]>}
#[derive(Clone,Debug,Default)]
/// tensor value dimensions
pub struct Shape{dims:Arc<[usize]>,strides:Arc<[usize]>}
#[derive(Clone,Debug,Default)]
/// nn layer io value. stores data and dimensions
pub struct Value{buffer:Arc<[f32]>,count:usize,offset:usize,shape:Shape}
use std::{
	ops::Deref,sync::Arc
};
