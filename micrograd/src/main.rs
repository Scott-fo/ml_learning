use micrograd::engine::{print_graph, Tensor};
use std::rc::Rc;

fn main() {
    let compute = || -> Rc<Tensor> {
        let x1 = Tensor::new(2.0, "x1");
        let x2 = Tensor::new(0.0, "x2");
        let w1 = Tensor::new(-3.0, "w1");
        let w2 = Tensor::new(1.0, "w2");
        let b = Tensor::new(6.8813735870195432, "b");
        let x1w1 = Tensor::mul(&x1, &w1);
        let x2w2 = Tensor::mul(&x2, &w2);
        let x1w1x2w2 = Tensor::add(&x1w1, &x2w2);
        let n = Tensor::add(&x1w1x2w2, &b);
        Tensor::tanh(&n)
    };

    let output = compute();
    Tensor::backward(&output);
    print_graph(&output, 0);
}
