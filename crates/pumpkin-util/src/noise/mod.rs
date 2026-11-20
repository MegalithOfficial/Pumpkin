pub mod perlin;
pub mod simplex;
pub mod volume;

/// A 3D gradient vector used for noise calculations.
pub struct Gradient {
    /// The X component of the gradient vector.
    x: f64,
    /// The Y component of the gradient vector.
    y: f64,
    /// The Z component of the gradient vector.
    z: f64,
}

const GRADIENT_VECTORS: [[f32; 3]; 16] = include!("perlin_gradients.in");

/// A pre-computed set of 16 gradient vectors for 3D noise generation.
pub const GRADIENTS: [Gradient; 16] = {
    let mut gradients = [const {
        Gradient {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }; GRADIENT_VECTORS.len()];
    let mut index = 0;
    while index < gradients.len() {
        let [x, y, z] = GRADIENT_VECTORS[index];
        gradients[index] = Gradient {
            x: x as f64,
            y: y as f64,
            z: z as f64,
        };
        index += 1;
    }
    gradients
};

impl Gradient {
    /// Computes the dot product of this gradient vector with the given coordinates.
    ///
    /// # Arguments
    /// - `x` – The X coordinate to dot with.
    /// - `y` – The Y coordinate to dot with.
    /// - `z` – The Z coordinate to dot with.
    ///
    /// # Returns
    /// The dot product `self.x * x + self.y * y + self.z * z`.
    #[inline]
    #[must_use]
    pub const fn dot(&self, x: f64, y: f64, z: f64) -> f64 {
        // When using mul_add without target-feature=+fma, you get a huge performance cost
        // because it lowers into a libm call 16x per Perlin sample.
        //
        // This improves performance by something crazy like 15%
        self.x * x + self.y * y + self.z * z
    }
}
