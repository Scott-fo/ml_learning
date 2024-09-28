use std::{
    cell::RefCell,
    collections::HashSet,
    fmt::Display,
    ops::{Add, Mul},
    rc::Rc,
};

#[derive(Clone)]
enum Op {
    Add(Tensor, Tensor),
    Mul(Tensor, Tensor),
    Tanh(Tensor),
    Exp(Tensor),
}

#[derive(Clone)]
pub struct Value {
    label: String,
    data: f64,
    grad: RefCell<f64>,
    op: Option<Op>,
}

#[derive(Clone)]
pub struct Tensor(Rc<Value>);

impl Add for Tensor {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let label = format!("{}+{}", self.0.label, rhs.0.label);

        Tensor::with_op(
            self.0.data + rhs.0.data,
            Some(Op::Add(self.clone(), rhs.clone())),
            label,
        )
    }
}

impl Add<Tensor> for f64 {
    type Output = Tensor;

    fn add(self, rhs: Tensor) -> Self::Output {
        let label = format!("{}+{}", rhs.0.label, self);

        Tensor::with_op(
            rhs.0.data + self,
            Some(Op::Add(Tensor::new(self, &self.to_string()), rhs.clone())),
            label,
        )
    }
}

impl Add<f64> for Tensor {
    type Output = Self;

    fn add(self, rhs: f64) -> Self::Output {
        rhs + self
    }
}

impl Mul for Tensor {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let label = format!("{}*{}", self.0.label, rhs.0.label);

        Tensor::with_op(
            self.0.data * rhs.0.data,
            Some(Op::Mul(self.clone(), rhs.clone())),
            label,
        )
    }
}

impl Mul<Tensor> for f64 {
    type Output = Tensor;

    fn mul(self, rhs: Tensor) -> Self::Output {
        let label = format!("{}*{}", rhs.0.label, self);

        Tensor::with_op(
            rhs.0.data * self,
            Some(Op::Add(Tensor::new(self, &self.to_string()), rhs.clone())),
            label,
        )
    }
}

impl Mul<f64> for Tensor {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        rhs * self
    }
}

impl Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Tensor(label={}, data={}, grad={})",
            self.0.label,
            self.0.data,
            self.0.grad.borrow()
        )
    }
}

impl Tensor {
    pub fn new(data: f64, label: &str) -> Tensor {
        Tensor(Rc::new(Value {
            label: label.to_string(),
            data,
            grad: RefCell::new(0.0),
            op: None,
        }))
    }

    fn with_op(data: f64, op: Option<Op>, label: String) -> Tensor {
        Tensor(Rc::new(Value {
            data,
            op,
            label: label.to_string(),
            grad: RefCell::new(0.0),
        }))
    }

    pub fn backward(self: &Self) {
        let mut topo: Vec<Tensor> = Vec::new();
        let mut visited: HashSet<*const Tensor> = HashSet::new();

        fn build_topo(v: &Tensor, topo: &mut Vec<Tensor>, visited: &mut HashSet<*const Tensor>) {
            let ptr = v as *const Tensor;

            if !visited.contains(&ptr) {
                visited.insert(ptr);
                if let Some(op) = &v.0.op {
                    match op {
                        Op::Add(t1, t2) | Op::Mul(t1, t2) => {
                            build_topo(&t1, topo, visited);
                            build_topo(&t2, topo, visited);
                        }
                        Op::Tanh(t) | Op::Exp(t) => build_topo(&t, topo, visited),
                    }
                }
                topo.push(v.clone());
            }
        }

        build_topo(self, &mut topo, &mut visited);

        *self.0.grad.borrow_mut() = 1.0;

        for v in topo.iter().rev() {
            if let Some(op) = &v.0.op {
                let current_grad = *v.0.grad.borrow();

                match op {
                    Op::Add(t1, t2) => {
                        *t1.0.grad.borrow_mut() += current_grad;
                        *t2.0.grad.borrow_mut() += current_grad;
                    }
                    Op::Mul(t1, t2) => {
                        *t1.0.grad.borrow_mut() += t2.0.data * current_grad;
                        *t2.0.grad.borrow_mut() += t1.0.data * current_grad;
                    }
                    Op::Tanh(t) => {
                        let g = (1.0 - (v.0.data * v.0.data)) * current_grad;
                        *t.0.grad.borrow_mut() += g;
                    }
                    Op::Exp(t) => {
                        *t.0.grad.borrow_mut() += v.0.data * current_grad;
                    }
                }
            }
        }
    }

    pub fn tanh(self: &Self) -> Self {
        let label = format!("tanh({})", self.0.label);
        Self::with_op(self.0.data.tanh(), Some(Op::Tanh(self.clone())), label)
    }

    pub fn exp(self: &Self) -> Self {
        let label = format!("exp({})", self.0.label);
        Self::with_op(self.0.data.exp(), Some(Op::Exp(self.clone())), label)
    }

    pub fn print_graph(self: &Self, depth: usize) {
        let indent = " ".repeat(depth * 2);
        println!("{}{}", indent, self);

        if let Some(op) = &self.0.op {
            match op {
                Op::Add(t1, t2) | Op::Mul(t1, t2) => {
                    Self::print_graph(t1, depth + 1);
                    Self::print_graph(t2, depth + 1);
                }
                Op::Tanh(t) | Op::Exp(t) => Self::print_graph(t, depth + 1),
            }
        }
    }
}
