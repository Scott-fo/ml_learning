use core::panic;
use std::{
    cell::RefCell,
    collections::HashSet,
    fmt::Display,
    ops::{Add, Mul},
    rc::Rc,
};

#[derive(Clone)]
pub enum Op {
    Add,
    Mul,
    Tanh,
}

#[derive(Clone)]
pub struct Tensor {
    pub data: f64,
    pub grad: RefCell<f64>,
    pub op: Option<Op>,
    pub prev: Vec<Rc<Tensor>>,
}

impl Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor(data={}, grad={})", self.data, self.grad.borrow())
    }
}

impl Tensor {
    pub fn new(data: f64) -> Self {
        Self {
            data,
            grad: RefCell::new(0.0),
            prev: vec![],
            op: None,
        }
    }

    pub fn backward(self: &Self) {
        if let Some(op) = &self.op {
            let current_grad = *self.grad.borrow();

            match op {
                Op::Add => {
                    if self.prev.len() != 2 {
                        panic!("only handling length of 2 for now");
                    }

                    *self.prev[0].grad.borrow_mut() = current_grad * 1.0;
                    *self.prev[1].grad.borrow_mut() = current_grad * 1.0;
                }
                Op::Mul => {
                    if self.prev.len() != 2 {
                        panic!("only handling length of 2 for now");
                    }

                    *self.prev[0].grad.borrow_mut() = self.prev[1].data * current_grad;
                    *self.prev[1].grad.borrow_mut() = self.prev[0].data * current_grad;
                }
                Op::Tanh => {
                    if self.prev.len() != 1 {
                        panic!("expected tanh to only have 1 prev");
                    }

                    let g = (1.0 - (self.data * self.data)) * current_grad;
                    *self.prev[0].grad.borrow_mut() = g;
                }
            }
        }
    }

    pub fn tanh(self: Self) -> Self {
        let th = self.data.tanh();

        Self {
            data: th,
            grad: RefCell::new(0.0),
            op: Some(Op::Tanh),
            prev: vec![Rc::new(self)],
        }
    }

    pub fn prev_to_string(&self) -> String {
        let inner = self
            .prev
            .iter()
            .map(|tensor| tensor.to_string())
            .collect::<Vec<_>>()
            .join(", ");

        format!("[{}]", inner)
    }
}

impl Add for Tensor {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let sum = self.data + rhs.data;

        Self {
            data: sum,
            grad: RefCell::new(0.0),
            op: Some(Op::Add),
            prev: vec![Rc::new(self), Rc::new(rhs)],
        }
    }
}

impl Mul for Tensor {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let prod = self.data * rhs.data;

        Self {
            data: prod,
            grad: RefCell::new(0.0),
            op: Some(Op::Mul),
            prev: vec![Rc::new(self), Rc::new(rhs)],
        }
    }
}
