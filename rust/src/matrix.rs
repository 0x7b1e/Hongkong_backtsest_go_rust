//! Dense row-major matrices shaped ``(bars, symbols)``.

#[derive(Debug, Clone, PartialEq)]
pub struct Mat<T> {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<T>,
}

impl<T: Copy> Mat<T> {
    pub fn filled(rows: usize, cols: usize, value: T) -> Self {
        Mat {
            rows,
            cols,
            data: vec![value; rows * cols],
        }
    }

    #[inline]
    pub fn get(&self, row: usize, col: usize) -> T {
        self.data[row * self.cols + col]
    }

    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: T) {
        self.data[row * self.cols + col] = value;
    }

    #[inline]
    pub fn row(&self, row: usize) -> &[T] {
        &self.data[row * self.cols..(row + 1) * self.cols]
    }

    #[inline]
    pub fn row_mut(&mut self, row: usize) -> &mut [T] {
        let cols = self.cols;
        &mut self.data[row * cols..(row + 1) * cols]
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
}

/// Sequential dot product with NaN propagation like ``np.dot``.
#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    pairwise_sum_map(a.len(), |i| a[i] * b[i])
}

/// NumPy-style pairwise summation of ``values``.
pub fn sum(values: &[f64]) -> f64 {
    pairwise_sum_map(values.len(), |i| values[i])
}

/// Pairwise summation following NumPy's ``pairwise_sum`` blocking so that
/// reductions agree with ``ndarray.sum`` to the last bit whenever possible.
pub fn pairwise_sum_map<F: Fn(usize) -> f64 + Copy>(n: usize, f: F) -> f64 {
    pairwise(0, n, f)
}

fn pairwise<F: Fn(usize) -> f64 + Copy>(start: usize, n: usize, f: F) -> f64 {
    const PW_BLOCKSIZE: usize = 128;
    if n < 8 {
        let mut res = 0.0;
        // NumPy starts from -0.0 for an empty sum but adds sequentially.
        for i in 0..n {
            res += f(start + i);
        }
        res
    } else if n <= PW_BLOCKSIZE {
        let mut r = [0.0f64; 8];
        for (k, slot) in r.iter_mut().enumerate() {
            *slot = f(start + k);
        }
        let mut i = 8;
        while i < n - (n % 8) {
            for (k, slot) in r.iter_mut().enumerate() {
                *slot += f(start + i + k);
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += f(start + i);
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise(start, n2, f) + pairwise(start + n2, n - n2, f)
    }
}
