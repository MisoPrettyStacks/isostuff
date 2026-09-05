import { LatentDimensionAgent } from './base-agent.js';

/**
 * Entropy Momentum Agent
 * Rides directional trends when the book transitions into highly organized, low-entropy structures.
 */
export class EntropyMomentumAgent extends LatentDimensionAgent {
    act(z, thermo) {
        let action = 0.0;
        
        // High free energy + cold temperature indicates non-equilibrium directional pressure
        if (thermo.freeEnergy > 10.0 && thermo.temperature < 1.0) {
            action = Math.sign(z[this.focusDim]) * Math.min(Math.abs(z[this.focusDim]), 1.0);
        }
        
        return {
            agent: this.name,
            action: action,
            conviction: Math.abs(action)
        };
    }
}
