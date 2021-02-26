# LQE - Linear Quadratic Estimator (Kalman Filter)

A naive implementation of a Kalman Filter in Rust.

## Usage

```rust
let lqe = LQE {
    measurement: 3.0,
    variance: 2.0
};

assert_eq!(lqe.next(5.0, 3.0).result(), (6.125, 3.0));
assert_eq!(lqe.next(5.0, 3.0).next(7.0, 1.0).result(), (8.225, 2.625));
```