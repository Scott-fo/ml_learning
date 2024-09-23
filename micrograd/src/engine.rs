use std::{cell::RefCell, collections::HashSet, fmt::Display, rc::Rc};

#[derive(Clone)]
pub enum Op {
    Add(Rc<Tensor>, Rc<Tensor>),
    Mul(Rc<Tensor>, Rc<Tensor>),
    Tanh(Rc<Tensor>),
}

#[derive(Clone)]
pub struct Tensor {
    pub label: String,
    pub data: f64,
    pub grad: RefCell<f64>,
    pub op: Option<Op>,
}

impl Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Tensor(label={}, data={}, grad={})",
            self.label,
            self.data,
            self.grad.borrow()
        )
    }
}

impl Tensor {
    pub fn new(data: f64, label: &str) -> Rc<Self> {
        Rc::new(Self {
            label: label.to_string(),
            data,
            grad: RefCell::new(0.0),
            op: None,
        })
    }

    pub fn backward(self: &Rc<Self>) {
        let mut topo: Vec<Rc<Tensor>> = Vec::new();
        let mut visited: HashSet<*const Tensor> = HashSet::new();

        fn build_topo(
            v: &Rc<Tensor>,
            topo: &mut Vec<Rc<Tensor>>,
            visited: &mut HashSet<*const Tensor>,
        ) {
            if !visited.contains(&(Rc::as_ptr(v) as *const Tensor)) {
                visited.insert(Rc::as_ptr(v) as *const Tensor);
                if let Some(op) = &v.op {
                    match op {
                        Op::Add(t1, t2) | Op::Mul(t1, t2) => {
                            build_topo(t1, topo, visited);
                            build_topo(t2, topo, visited);
                        }
                        Op::Tanh(t) => build_topo(t, topo, visited),
                    }
                }
                topo.push(Rc::clone(v));
            }
        }

        build_topo(self, &mut topo, &mut visited);

        *self.grad.borrow_mut() = 1.0;

        for v in topo.iter().rev() {
            if let Some(op) = &v.op {
                let current_grad = *v.grad.borrow();
                match op {
                    Op::Add(t1, t2) => {
                        *t1.grad.borrow_mut() += current_grad;
                        *t2.grad.borrow_mut() += current_grad;
                    }
                    Op::Mul(t1, t2) => {
                        *t1.grad.borrow_mut() += t2.data * current_grad;
                        *t2.grad.borrow_mut() += t1.data * current_grad;
                    }
                    Op::Tanh(t) => {
                        let g = (1.0 - (v.data * v.data)) * current_grad;
                        *t.grad.borrow_mut() += g;
                    }
                }
            }
        }
    }

    pub fn tanh(self: &Rc<Self>) -> Rc<Tensor> {
        let label = format!("tanh({})", self.label);

        Rc::new(Tensor {
            label,
            data: self.data.tanh(),
            grad: RefCell::new(0.0),
            op: Some(Op::Tanh(Rc::clone(self))),
        })
    }

    pub fn print_graph(self: &Rc<Self>, depth: usize) {
        let indent = " ".repeat(depth * 2);
        println!("{}{}", indent, self);
        if let Some(op) = &self.op {
            match op {
                Op::Add(t1, t2) | Op::Mul(t1, t2) => {
                    Self::print_graph(t1, depth + 1);
                    Self::print_graph(t2, depth + 1);
                }
                Op::Tanh(t) => Self::print_graph(t, depth + 1),
            }
        }
    }
}

pub fn add(t1: &Rc<Tensor>, t2: &Rc<Tensor>) -> Rc<Tensor> {
    let label = format!("{}+{}", t1.label, t2.label);

    Rc::new(Tensor {
        label,
        data: t1.data + t2.data,
        grad: RefCell::new(0.0),
        op: Some(Op::Add(Rc::clone(t1), Rc::clone(t2))),
    })
}

pub fn mul(t1: &Rc<Tensor>, t2: &Rc<Tensor>) -> Rc<Tensor> {
    let label = format!("{}*{}", t1.label, t2.label);

    Rc::new(Tensor {
        label,
        data: t1.data * t2.data,
        grad: RefCell::new(0.0),
        op: Some(Op::Mul(Rc::clone(t1), Rc::clone(t2))),
    })
}
