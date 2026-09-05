import * as tf from '@tensorflow/tfjs';

// ============================================================================
// THERMODYNAMICALLY REGULARIZED VAE IN TENSORFLOW.JS
// ============================================================================

class LatentDiscoveryTFVAE {
    /**
     * @param {number} inputDim - Number of order book features (e.g., 40)
     * @param {number} latentDim - Hidden dimensions to discover (e.g., 3)
     */
    constructor(inputDim = 40, latentDim = 3) {
        this.inputDim = inputDim;
        this.latentDim = latentDim;
        this.buildArchitecture();
    }

    buildArchitecture() {
        // --- Encoder Network ---
        const encoderInput = tf.input({ shape: [this.inputDim] });
        let h = tf.layers.dense({ units: 128, activation: 'silu' }).apply(encoderInput);
        h = tf.layers.layerNormalization().apply(h);
        h = tf.layers.dense({ units: 64, activation: 'silu' }).apply(h);
        
        const mu = tf.layers.dense({ units: this.latentDim, name: 'mu' }).apply(h);
        const logVar = tf.layers.dense({ units: this.latentDim, name: 'logvar' }).apply(h);
        
        this.encoder = tf.model({ inputs: encoderInput, outputs: [mu, logVar] });

        // --- Decoder Network ---
        const decoderInput = tf.input({ shape: [this.latentDim] });
        let d = tf.layers.dense({ units: 64, activation: 'silu' }).apply(decoderInput);
        d = tf.layers.dense({ units: 128, activation: 'silu' }).apply(d);
        const reconX = tf.layers.dense({ units: this.inputDim }).apply(d);
        
        this.decoder = tf.model({ inputs: decoderInput, outputs: reconX });
    }

    /**
     * Reparameterization Trick: z = mu + sigma * epsilon
     */
    reparameterize(mu, logVar) {
        return tf.tidy(() => {
            const eps = tf.randomNormal(mu.shape);
            const sigma = tf.exp(logVar.mul(0.5));
            return mu.add(eps.mul(sigma));
        });
    }

    /**
     * Executes a forward training or inference pass.
     * @param {tf.Tensor2D} x - Raw input order book snapshots tensor
     */
    forward(x) {
        return tf.tidy(() => {
            const [mu, logVar] = this.encoder.apply(x);
            const z = this.reparameterize(mu, logVar);
            const reconX = this.decoder.apply(z);
            return { reconX, z, mu, logVar };
        });
    }

    /**
     * Custom Physics-Informed VAE Loss Function
     */
    calculateLoss(x, reconX, mu, logVar, avgTemperature) {
        return tf.tidy(() => {
            // Reconstruction Accuracy (MSE)
            const reconLoss = tf.losses.meanSquaredError(x, reconX);

            // Kullback-Leibler Divergence
            const kld = tf.mean(
                tf.sum(
                    tf.scalar(1.0)
                        .add(logVar)
                        .sub(tf.square(mu))
                        .sub(tf.exp(logVar)),
                    1
                ).mul(-0.5)
            );

            // Thermodynamic Variance Consistency: Hidden energy variance scales with market temp
            const latentVariance = tf.moments(mu, 0).variance.sum();
            const thermoTarget = tf.scalar(avgTemperature);
            const thermoPenalty = tf.losses.meanSquaredError(thermoTarget, latentVariance);

            // Total integrated structural loss
            return reconLoss.add(kld.mul(0.01)).add(thermoPenalty.mul(0.1));
        });
    }
}

// ============================================================================
// PIPELINE VERIFICATION EXECUTION
// ============================================================================
(async () => {
    // Wait for the WebGL/WASM backend to initialize
    await tf.ready();
    console.log(`Using active computational backend: ${tf.getBackend()}`);

    const batchSize = 4;
    const inputDim = 40;
    const latentDim = 3;

    const vaeEngine = new LatentDiscoveryTFVAE(inputDim, latentDim);
    
    // Simulate fake market data snapshot batch tensor
    const simulatedMarketBatch = tf.randomUniform([batchSize, inputDim]);
    const mockAverageTemperature = 0.42;

    // Run forward computational graph execution
    const outputs = vaeEngine.forward(simulatedMarketBatch);
    const executionLoss = vaeEngine.calculateLoss(
        simulatedMarketBatch, 
        outputs.reconX, 
        outputs.mu, 
        outputs.logVar, 
        mockAverageTemperature
    );

    executionLoss.print(); // Displays training optimization step loss

    // Memory cleanup metrics verification
    tf.dispose([simulatedMarketBatch, outputs.reconX, outputs.z, outputs.mu, outputs.logVar, executionLoss]);
})();
