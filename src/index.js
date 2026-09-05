import * as tf from '@tensorflow/tfjs';
import { ThermodynamicLiquidityLayer } from './physics/thermodynamic-layer.js';
import { LatentDiscoveryTFVAE } from './ml/vae-engine.js';
import { calculatePhysicsInformedLoss } from './ml/loss.js';
import { NonEquilibriumArbitrageur } from './agents/arbitrage-agent.js';
import { EntropyMomentumAgent } from './agents/momentum-agent.js';
import { BarrierLiquidityProvider } from './agents/liquidity-agent.js';
import { MultiAgentSynthesizer } from './agents/synthesizer.js';
import { SyntheticOrderBookGenerator } from './utils/orderbook-generator.js';

async function executeFrameworkPipeline() {
    console.log('--- Initializing Latent Thermo-Trading Framework ---');
    await tf.ready();
    console.log(`Compute Engine Backend: ${tf.getBackend()}\n`);

    // 1. Pipeline Layout Parameters
    const levels = 10;
    const inputDim = 4 * levels;
    const latentDim = 3;

    // 2. Initialize Structural Components
    const dataGenerator = new SyntheticOrderBookGenerator(levels);
    const physicsLayer = new ThermodynamicLiquidityLayer(levels, 1.0);
    const vaeModel = new LatentDiscoveryTFVAE(inputDim, latentDim);
    
    // Configure standard optimizer for network tuning
    const modelOptimizer = tf.train.adam(0.001);

    const tradingAgents = [
        new NonEquilibriumArbitrageur('NonEq_Arb', 0),
        new EntropyMomentumAgent('Entropy_Mom', 1),
        new BarrierLiquidityProvider('Barrier_LP', 2)
    ];
    const orchestrator = new MultiAgentSynthesizer(tradingAgents);

    // 3. Collect Raw Level-2 Snapshot Profiles
    const rawDataBatch = dataGenerator.generateBatch(4);
    
    // 4. Compute Microstructure Thermodynamics
    const thermoStates = physicsLayer.forward(rawDataBatch);
    const averageBatchTemperature = thermoStates.reduce((sum, s) => sum + s.temperature, 0) / thermoStates.length;

    // 5. Execute Machine Learning Forward Pass and Gradient Minimization
    let computedLossValue = 0;
    const inputTensor2D = tf.tensor2d(rawDataBatch);

    // Wrap the operations in a variable gradient tracking loop
    const optimizationCost = modelOptimizer.minimize(() => {
        const networkOutputs = vaeModel.forward(inputTensor2D);
        const coreLossScalar = calculatePhysicsInformedLoss(
            inputTensor2D,
            networkOutputs.reconX,
            networkOutputs.mu,
            networkOutputs.logVar,
            averageBatchTemperature
        );
        
        computedLossValue = coreLossScalar.dataSync()[0];
        return coreLossScalar;
    }, true);

    // Extract current continuous hidden coordinates from the updated state space
    const structuralOutputs = vaeModel.forward(inputTensor2D);
    const latentCoordinatesZ = await structuralOutputs.z.array();

    // 6. Run Multi-Agent Orchestration Across Discovered Dimensions
    const consensusDecisions = orchestrator.orchestrate(latentCoordinatesZ, thermoStates);

    // 7. System Diagnostic Reporting
    console.log(`[Optimization] Structural Convergence Loss Matrix Value: ${computedLossValue.toFixed(6)}`);
    console.log(`[Optimization] Average Local Microstructure Temperature: ${averageBatchTemperature.toFixed(4)}\n`);
    
    console.log('--- Multi-Agent Execution Grid Actions ---');
    consensusDecisions.forEach((decision, index) => {
        console.log(
            `Snapshot [${index}] -> Consensus Order Direction: ${decision.consensusAction >= 0 ? '+' : ''}${decision.consensusAction.toFixed(4)} ` +
            `| Core Spread Barrier: ${decision.energyBarrier.toFixed(4)} ` +
            `| Temp Value: ${decision.temperature.toFixed(4)}`
        );
    });

    // 8. Explicit memory allocations housecleaning
    tf.dispose([inputTensor2D, optimizationCost, structuralOutputs.reconX, structuralOutputs.z, structuralOutputs.mu, structuralOutputs.logVar]);
}

// Fire system thread execution loop
executeFrameworkPipeline().catch(error => console.error('Critical Pipeline Execution Failure:', error));
