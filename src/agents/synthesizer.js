/**
 * Resolves conflicting multi-agent votes using conviction-weighted consensus.
 */
export class MultiAgentSynthesizer {
    /**
     * @param {Array<LatentDimensionAgent>} agents - Collection of initialized multi-coordinate agents.
     */
    constructor(agents) {
        this.agents = agents;
    }

    /**
     * Harmonizes agent decisions against corresponding thermodynamic contexts across a batch.
     * @param {number[][]} zBatch - Raw array representation of the discovered latent space.
     * @param {Array<Object>} thermoBatch - Array of parsed structural thermodynamic macro-states.
     * @returns {Array<Object>} Consensus target parameters for direct downstream book placement.
     */
    orchestrate(zBatch, thermoBatch) {
        return zBatch.map((zSample, i) => {
            const thermo = thermoBatch[i];
            let totalConviction = 0;
            let weightedPosition = 0;

            const agentDetails = this.agents.map(agent => {
                const decision = agent.act(zSample, thermo);
                weightedPosition += (decision.action * decision.conviction);
                totalConviction += decision.conviction;
                return decision;
            });

            // Conviction-weighted net trade position calculation
            const consensusAction = totalConviction > 0 ? (weightedPosition / totalConviction) : 0.0;

            return {
                consensusAction,
                energyBarrier: thermo.energyBarrier,
                temperature: thermo.temperature,
                agentDetails
            };
        });
    }
}
