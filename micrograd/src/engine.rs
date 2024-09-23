use std::{
    cell::RefCell,
    collections::HashSet,
    fmt::Display,
    ops::{Add, Mul},
    rc::Rc,
};

use ordered_float::OrderedFloat;

#[derive(Clone, Eq, PartialEq)]
enum Op {
    Add,
    Mul,
    Tanh,
}

#[derive(Eq, PartialEq, Clone)]
pub struct Tensor {
    pub data: OrderedFloat<f64>,

    grad: RefCell<OrderedFloat<f64>>,
    op: Option<Op>,
    prev: HashSet<Rc<Tensor>>,
}

impl std::hash::Hash for Tensor {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.data.hash(state);
        for tensor in &self.prev {
            std::ptr::hash(Rc::as_ptr(tensor), state);
        }
    }
}

impl Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor(data={})", self.data)
    }
}

impl Tensor {
    pub fn new(data: f64) -> Self {
        Self {
            data: OrderedFloat(data),
            grad: RefCell::new(OrderedFloat(0.0)),
            prev: HashSet::new(),
            op: None,
        }
    }

    pub fn tanh(self: Self) -> Self {
        let th = self.data.tanh();

        let mut prev = HashSet::new();
        prev.insert(Rc::new(self));

        Self {
            data: OrderedFloat(th),
            grad: RefCell::new(OrderedFloat(0.0)),
            op: Some(Op::Tanh),
            prev,
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

        let mut prev = HashSet::new();
        prev.insert(Rc::new(self));
        prev.insert(Rc::new(rhs));

        Self {
            data: sum,
            grad: RefCell::new(OrderedFloat(0.0)),
            op: Some(Op::Add),
            prev,
        }
    }
}

impl Mul for Tensor {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let prod = self.data * rhs.data;

        let mut c = HashSet::new();
        c.insert(Rc::new(self));
        c.insert(Rc::new(rhs));

        Self {
            data: prod,
            grad: RefCell::new(OrderedFloat(0.0)),
            op: Some(Op::Mul),
            prev: c,
        }
    }
}
