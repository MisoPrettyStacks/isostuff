import * as tf from '@tensorflow/tfjs';

/**
 * Computes the integrated physics-informed loss for the Latent Discovery Engine.
 * 
 * L = MeanSquaredError(x, reconX) 
 *     + beta * KL_Divergence(mu, logVar)
 *     + gamma * MSE(Variance(mu), avgTemperature)
 * 
 * @param {tf.Tensor2D} x - Original Level-2 order book input tensor.
 * @param {tf.Tensor2D} reconX - Reconstructed output tensor from the decoder.
 * @param {tf.Tensor2D} mu - Latent mean vector tensor.
 * @param {tf.Tensor2D} logVar - Latent log-variance vector tensor.
 * @param {number} avgTemperature - Average microscale temperature computed via the physics layer.
 * @param {number} beta - Weight parameter for the KL Divergence regularizer (default: 0.01).
 * @param {number} gamma - Weight parameter for the thermodynamic alignment penalty (default: 0.1).
 * @returns {tf.Scalar} Single scalar loss value optimized for gradient descent.
 */
export function calculatePhysicsInformedLoss(x, reconX, mu, logVar, avgTemperature, beta = 0.01, gamma = 0.1) {
    return tf.tidy(() => {
        // 1. Reconstruction Accuracy Loss (MSE across market profiles)
        const reconLoss = tf.losses.meanSquaredError(x, reconX);

        // 2. Kullback-Leibler Divergence (Enforces latent space Gaussian normality)
        const kld = tf.mean(
            tf.sum(
                tf.scalar(1.0)
                    .add(logVar)
                    .sub(tf.square(mu))
                    .sub(tf.exp(logVar)),
                1
            ).mul(-0.5)
        );

        // 3. Thermodynamic Variance Consistency Regularizer
        // Calculates the empirical variance of the latent dimensions across the batch
        const latentVariance = tf.moments(mu, 0).variance.sum();
        const thermoTarget = tf.scalar(avgTemperature);
        const thermoPenalty = tf.losses.meanSquaredError(thermoTarget, latentVariance);

        // 4. Synthesize all components into a scalar objective function
        return reconLoss.add(kld.mul(beta)).add(thermoPenalty.mul(gamma));
    });
}
