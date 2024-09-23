use micrograd::engine;
use ordered_float::OrderedFloat;

fn main() {
    //let h = 0.0001;

    let compute = || -> OrderedFloat<f64> {
        let x1 = engine::Tensor::new(2.0);
        let x2 = engine::Tensor::new(0.0);

        let w1 = engine::Tensor::new(-3.0);
        let w2 = engine::Tensor::new(1.0);

        let b = engine::Tensor::new(6.7);

        let x1w1 = x1 * w1;
        let x2w2 = x2 * w2;
        let x1w1x2w2 = x1w1 + x2w2;

        let n = x1w1x2w2 + b;
        let o = n.tanh();
        o.data
    };

    println!("Got data: {}", compute());
}
