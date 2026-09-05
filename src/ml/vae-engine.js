import * as tf from '@tensorflow/tfjs';

/**
 * Latent Discovery Engine: A Deep Variational Autoencoder implemented via TensorFlow.js
 * that compresses Level-2 microstructure profiles into continuous 'Undiscovered Dimensions'.
 */
export class LatentDiscoveryTFVAE {
    /**
     * @param {number} inputDim - Total input dimensions from order book mapping (e.g., 40)
     * @param {number} latentDim - Dimensions of the hidden orthogonal space Z (e.g., 3)
     */
    constructor(inputDim = 40, latentDim = 3) {
        this.inputDim = inputDim;
        this.latentDim = latentDim;
        this.buildArchitecture();
    }

    /**
     * Builds and configures the standard Encoder and Decoder sequential networks.
     */
    buildArchitecture() {
        // --- Encoder sub-network (q_phi(z|x)) ---
        const encoderInput = tf.input({ shape: [this.inputDim] });
        let h = tf.layers.dense({ units: 128, activation: 'silu' }).apply(encoderInput);
        h = tf.layers.layerNormalization().apply(h);
        h = tf.layers.dense({ units: 64, activation: 'silu' }).apply(h);
        
        const mu = tf.layers.dense({ units: this.latentDim, name: 'mu_layer' }).apply(h);
        const logVar = tf.layers.dense({ units: this.latentDim, name: 'logvar_layer' }).apply(h);
        
        this.encoder = tf.model({ inputs: encoderInput, outputs: [mu, logVar] });

        // --- Decoder sub-network (p_theta(x|z)) ---
        const decoderInput = tf.input({ shape: [this.latentDim] });
        let d = tf.layers.dense({ units: 64, activation: 'silu' }).apply(decoderInput);
        d = tf.layers.dense({ units: 128, activation: 'silu' }).apply(d);
        const reconX = tf.layers.dense({ units: this.inputDim, name: 'reconstruction_layer' }).apply(d);
        
        this.decoder = tf.model({ inputs: decoderInput, outputs: reconX });
    }

    /**
     * Reparameterization Trick: z = mu + sigma * epsilon
     * Allows backpropagation through stochastic latent bottlenecks.
     * @param {tf.Tensor2D} mu 
     * @param {tf.Tensor2D} logVar 
     * @returns {tf.Tensor2D} Latent coordinate space tensor Z
     */
    reparameterize(mu, logVar) {
        return tf.tidy(() => {
            const eps = tf.randomNormal(mu.shape);
            const sigma = tf.exp(logVar.mul(0.5));
            return mu.add(eps.mul(sigma));
        });
    }

    /**
     * Computes a full forward execution graph transformation.
     * @param {tf.Tensor2D} x - Level-2 snapshot sample batch tensor
     */
    forward(x) {
        return tf.tidy(() => {
            const [mu, logVar] = this.encoder.apply(x);
            const z = this.reparameterize(mu, logVar);
            const reconX = this.decoder.apply(z);
            return { reconX, z, mu, logVar };
        });
    }
}
