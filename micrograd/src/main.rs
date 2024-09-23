use micrograd::engine;
use std::rc::Rc;

fn main() {
    let compute = || -> engine::Tensor {
        let x1 = engine::Tensor::new(2.0);
        let x2 = engine::Tensor::new(0.0);

        let w1 = engine::Tensor::new(-3.0);
        let w2 = engine::Tensor::new(1.0);

        let b = engine::Tensor::new(6.8813735870195432);

        let x1w1 = x1 * w1;
        let x2w2 = x2 * w2;
        let x1w1x2w2 = x1w1 + x2w2;

        let n = x1w1x2w2 + b;
        n.tanh()
    };

    let output = compute();
    *output.grad.borrow_mut() = 1.0;
    output.backward();

    println!("Output: {}", output);
    println!("{}", output.prev_to_string());
}
