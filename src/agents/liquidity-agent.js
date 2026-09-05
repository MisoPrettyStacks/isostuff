import { LatentDimensionAgent } from './base-agent.js';

/**
 * Barrier Liquidity Provider
 * Captures spread yield when transition likelihood drops and volatility peaks at the edges.
 */
export class BarrierLiquidityProvider extends LatentDimensionAgent {
    act(z, thermo) {
        let action = 0.0;
        
        // Harvest spread when the energy barrier stretches and escape probability drops
        if (thermo.energyBarrier > 0.02 && thermo.transitionProbability < 0.2) {
            action = Math.max(-0.5, Math.min(0.5, z[this.focusDim])); // Controlled limit order placement
        }
        
        return {
            agent: this.name,
            action: action,
            conviction: Math.abs(action)
        };
    }
}
