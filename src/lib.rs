//! LQE - Linear Quadratic Estimator - Kalman Filter
//!
//! The LQE or Kalman filter is a recursive estimator. This means that only the estimated
//! state from the previous time step and the current measurement are needed to
//! compute the estimate for the current state.
//! [- Wikipedia](https://en.wikipedia.org/wiki/Kalman_filter)
//!
//! # Example:
//!
//! ```
//! use lqe::LQE;
//! let lqe = LQE {
//!   measurement: 7.0,
//!   variance: 2.0
//! };
//!
//! lqe.next(5.0, 3.0).next(7.0, 1.0).result();
//! // => (8.225, 2.625)
//! ```

/// LQE is a data type representing a single measurement with a variance or
/// confidence in that measurement.
///
/// `measurement` is the mean of the normal distribution. e.g `(100+150)/2 = 125.0`
///
/// `variance` is the standard inaccuracy +/- e.g `5.0`
///
/// # Example:
///
/// ```
/// use lqe::LQE;
/// let lqe = LQE {
///   measurement: 7.0,
///   variance: 2.0
/// };
/// ```
pub struct LQE {
    pub measurement: f64,
    pub variance: f64
}

impl LQE {
    /// `update` combines the past and current observation information to refine
    /// the state estimate.
    ///
    /// *Usually, you won't need to use this function manually but rather use the `next` function.*
    ///
    /// # Example:
    ///
    /// ```
    /// use lqe::LQE;
    /// let lqe = LQE { measurement: 7.0, variance: 2.0 };
    /// lqe.update(10.0, 2.0);
    /// // => (8.5, 5.0)
    /// ```
    pub fn update(&self, measurement: f64, variance: f64) -> (f64, f64) {
        // Calculate new measurement
        let a = &self.variance + variance;
        let c = (&self.measurement * variance) + (measurement * &self.variance);
        let m = (1.0 / &a) * c;
        // Calculate new variance
        let b = &self.variance * measurement;
        let z = b / a;
        (m, z)
    }

    /// `predict` uses the state estimate from the previous timestep to produce an
    /// estimate of the state at the current timestep.
    ///
    /// *Usually, you won't need to use this function manually but rather use the `next` function.*
    ///
    /// # Example:
    ///
    /// ```
    /// use lqe::LQE;
    /// let lqe = LQE { measurement: 7.0, variance: 2.0 };
    /// lqe.predict(10.0, 2.0);
    /// // => (17.0, 4.0)
    /// ```
    pub fn predict(&self, measurement: f64, variance: f64) -> (f64, f64) {
        let predicted_measurement = &self.measurement + measurement;
        let predicted_variance = &self.variance + variance;
        (predicted_measurement, predicted_variance)
    }

    /// `next` performs the entire predict - update cycle for a series of measurements.
    ///
    /// # Example:
    ///
    /// ```
    /// use lqe::LQE;
    /// let lqe = LQE { measurement: 3.0, variance: 2.0 };
    /// lqe.next(5.0, 3.0).result();
    /// // => (6.125, 3.0)
    /// ```
    pub fn next(&self, measurement: f64, variance: f64) -> LQE {
        let prediction = &self.predict(measurement, variance);
        let mid_filter = LQE {
            measurement: measurement,
            variance: variance
        };
        let updated_result = mid_filter.update(prediction.0, prediction.1);
        LQE {
            measurement: updated_result.0,
            variance: updated_result.1
        }
    }

    /// `result` returns the current state of the LQE as a tuple value.
    ///
    /// # Example:
    ///
    /// ```
    /// use lqe::LQE;
    /// let lqe = LQE { measurement: 7.0, variance: 2.0 };
    /// lqe.result();
    /// // => (7.0, 2.0)
    /// ```
    pub fn result (&self) -> (f64, f64) {
        (*&self.measurement, *&self.variance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn updates_from_measurements() {
        let lqe = LQE {
            measurement: 7.0,
            variance: 2.0
        };

        assert_eq!(lqe.update(10.0, 2.0), (8.5, 5.0));
    }

    #[test]
    fn predicts_next_value() {
        let lqe = LQE {
            measurement: 7.0,
            variance: 2.0
        };

        assert_eq!(lqe.predict(10.0, 2.0), (17.0, 4.0));
    }

    #[test]
    fn returns_result() {
        let lqe = LQE {
            measurement: 3.0,
            variance: 2.0
        };

        assert_eq!(lqe.result(), (3.0, 2.0))
    }

    #[test]
    fn runs_filter_correctly() {
        let lqe = LQE {
            measurement: 3.0,
            variance: 2.0
        };

      assert_eq!(lqe.next(5.0, 3.0).result(), (6.125, 3.0));
      assert_eq!(lqe.next(5.0, 3.0).next(7.0, 1.0).result(), (8.225, 2.625));
    }
}
