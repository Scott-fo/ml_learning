use micrograd::engine::Tensor;

fn main() {
    let compute = || -> Tensor {
        let x1 = Tensor::new(2.0, "x1");
        let x2 = Tensor::new(0.0, "x2");
        let w1 = Tensor::new(-3.0, "w1");
        let w2 = Tensor::new(1.0, "w2");
        let b = Tensor::new(6.8813735870195432, "b");

        let x1w1 = x1 * w1;
        let x2w2 = x2 * w2;
        let x1w1x2w2 = x1w1 + x2w2;

        let n = x1w1x2w2 + b;
        n.tanh()
    };

    /*
    let output = compute();
    output.backward();
    output.print_graph(0);
    */

    let a = Tensor::new(2.0, "a");
    let b = 2.0;

    println!("{}", b * a.clone());
    println!("{}", a.clone() * b);
    println!("{}", a.clone() + b);
    println!("{}", b + a.clone());
}
