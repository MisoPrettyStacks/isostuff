import { LatentDimensionAgent } from './base-agent.js';

/**
 * Non-Equilibrium Arbitrageur
 * Capitalizes on coordinate mean-reversion when transition boundaries are permeable.
 */
export class NonEquilibriumArbitrageur extends LatentDimensionAgent {
    act(z, thermo) {
        let action = 0.0;
        
        // Execute only if Kramers' escape rate indicates boundary permeability
        if (thermo.transitionProbability > 0.4) {
            action = -Math.tanh(z[this.focusDim]); // Mean-reversion vector counter-force
        }
        
        return {
            agent: this.name,
            action: action,
            conviction: Math.abs(action)
        };
    }
}
