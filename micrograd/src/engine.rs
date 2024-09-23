use std::{cell::RefCell, collections::HashSet, fmt::Display, rc::Rc};

#[derive(Clone)]
pub enum Op {
    Add,
    Mul,
    Tanh,
}

#[derive(Clone)]
pub struct Tensor {
    pub label: String,
    pub data: f64,
    pub grad: RefCell<f64>,
    pub op: Option<Op>,
    pub prev: Vec<Rc<Tensor>>,
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
    pub fn new(data: f64, label: &str) -> Rc<Tensor> {
        Rc::new(Self {
            label: label.to_string(),
            data,
            grad: RefCell::new(0.0),
            prev: vec![],
            op: None,
        })
    }

    pub fn add(t1: &Rc<Tensor>, t2: &Rc<Tensor>) -> Rc<Tensor> {
        let label = format!("{}+{}", t1.label, t2.label);

        Rc::new(Tensor {
            label,
            data: t1.data + t2.data,
            grad: RefCell::new(0.0),
            op: Some(Op::Add),
            prev: vec![Rc::clone(t1), Rc::clone(t2)],
        })
    }

    pub fn mul(t1: &Rc<Tensor>, t2: &Rc<Tensor>) -> Rc<Tensor> {
        let label = format!("{}*{}", t1.label, t2.label);

        Rc::new(Tensor {
            label,
            data: t1.data * t2.data,
            grad: RefCell::new(0.0),
            op: Some(Op::Mul),
            prev: vec![Rc::clone(t1), Rc::clone(t2)],
        })
    }

    pub fn tanh(t: &Rc<Tensor>) -> Rc<Tensor> {
        let label = format!("tanh({})", t.label);

        Rc::new(Tensor {
            label,
            data: t.data.tanh(),
            grad: RefCell::new(0.0),
            op: Some(Op::Tanh),
            prev: vec![Rc::clone(t)],
        })
    }

    pub fn backward(v: &Rc<Tensor>) {
        let mut topo: Vec<Rc<Tensor>> = Vec::new();
        let mut visited: HashSet<*const Tensor> = HashSet::new();

        fn build_topo(
            v: &Rc<Tensor>,
            topo: &mut Vec<Rc<Tensor>>,
            visited: &mut HashSet<*const Tensor>,
        ) {
            if !visited.contains(&(Rc::as_ptr(v) as *const Tensor)) {
                visited.insert(Rc::as_ptr(v) as *const Tensor);
                for child in &v.prev {
                    build_topo(child, topo, visited);
                }
                topo.push(Rc::clone(v));
            }
        }

        build_topo(v, &mut topo, &mut visited);

        *v.grad.borrow_mut() = 1.0;

        for v in topo.iter().rev() {
            if let Some(op) = &v.op {
                let current_grad = *v.grad.borrow();
                match op {
                    Op::Add => {
                        if v.prev.len() != 2 {
                            panic!("only handling length of 2 for now");
                        }
                        *v.prev[0].grad.borrow_mut() += current_grad * 1.0;
                        *v.prev[1].grad.borrow_mut() += current_grad * 1.0;
                    }
                    Op::Mul => {
                        if v.prev.len() != 2 {
                            panic!("only handling length of 2 for now");
                        }
                        *v.prev[0].grad.borrow_mut() += v.prev[1].data * current_grad;
                        *v.prev[1].grad.borrow_mut() += v.prev[0].data * current_grad;
                    }
                    Op::Tanh => {
                        if v.prev.len() != 1 {
                            panic!("expected tanh to only have 1 prev");
                        }
                        let g = (1.0 - (v.data * v.data)) * current_grad;
                        *v.prev[0].grad.borrow_mut() += g;
                    }
                }
            }
        }
    }
}

pub fn print_graph(tensor: &Rc<Tensor>, depth: usize) {
    let indent = " ".repeat(depth * 2);
    println!("{}{}", indent, tensor);
    for child in &tensor.prev {
        print_graph(child, depth + 1);
    }
}
