import test from 'node:test';
import assert from 'node:assert';
import * as tf from '@tensorflow/tfjs';
import { ThermodynamicLiquidityLayer } from '../src/physics/thermodynamic-layer.js';
import { LatentDiscoveryTFVAE } from '../src/ml/vae-engine.js';
import { SyntheticOrderBookGenerator } from '../src/utils/orderbook-generator.js';

test('Thermodynamic Layer Invariant Validation', async (t) => {
    await tf.ready();
    const generator = new SyntheticOrderBookGenerator(10);
    const physics = new ThermodynamicLiquidityLayer(10, 1.0);
    
    const mockBatch = generator.generateBatch(2);
    const results = physics.forward(mockBatch);

    assert.strictEqual(results.length, 2, 'Should process exactly 2 snapshot structures');
    
    results.forEach((state) => {
        assert.ok(state.temperature > 0, 'Microstructure temperature must resolve to a positive non-zero value');
        assert.ok(state.energyBarrier > 0, 'Friction energy potential barrier must be positive');
        assert.ok(state.transitionProbability >= 0 && state.transitionProbability <= 1, 'Escape probabilities must stay inside valid [0, 1] bounds');
    });
});

test('Latent Discovery VAE Tensor Dimensionality Integrity', async (t) => {
    const inputDim = 40;
    const latentDim = 3;
    const vae = new LatentDiscoveryTFVAE(inputDim, latentDim);
    
    const testInput = tf.randomUniform([4, inputDim]);
    const outputs = vae.forward(testInput);

    assert.deepStrictEqual(outputs.reconX.shape, [4, inputDim], 'Decoder output dimensions must match incoming Level-2 footprint vectors');
    assert.deepStrictEqual(outputs.z.shape, [4, latentDim], 'Discovered latent state space dimensions must match target coordinates count');

    tf.dispose([testInput, outputs.reconX, outputs.z, outputs.mu, outputs.logVar]);
});
