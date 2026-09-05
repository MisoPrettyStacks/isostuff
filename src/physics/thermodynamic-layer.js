/**
 * Maps Level-2 Order Book snapshots to non-equilibrium thermodynamic invariants.
 */
export class ThermodynamicLiquidityLayer {
    /**
     * @param {number} numLevels - Number of pricing depth levels provided in the snapshot.
     * @param {number} boltzmannConst - Scaling parameter k_B for escape transformations.
     */
    constructor(numLevels = 10, boltzmannConst = 1.0) {
        this.numLevels = numLevels;
        this.kB = boltzmannConst;
    }

    /**
     * Processes a collection of order books to extract thermodynamic invariants.
     * @param {number[][]} orderBookBatch - Shape: [batch_size, 4 * num_levels]
     *        Format per row: [bid_prices..., bid_sizes..., ask_prices..., ask_sizes...]
     * @returns {Array<Object>} Batch array of parsed thermodynamic macro-states.
     */
    forward(orderBookBatch) {
        return orderBookBatch.map(row => {
            const num = this.numLevels;
            
            // Extract top of book parameters
            const highestBid = row[0];
            const lowestAsk = row[2 * num];

            // 1. Calculate Internal Energy (U) as total immediate mass/volume
            let totalVol = 0;
            for (let l = 0; l < num; l++) {
                totalVol += row[num + l];     // Bid depth sizes
                totalVol += row[3 * num + l]; // Ask depth sizes
            }
            totalVol = Math.max(totalVol, 1e-8);

            // 2. Calculate Microstructure Entropy H(p) of layout allocations
            let entropy = 0;
            for (let l = 0; l < num; l++) {
                const bProb = row[num + l] / totalVol;
                const aProb = row[3 * num + l] / totalVol;
                if (bProb > 1e-8) entropy -= bProb * Math.log(bProb);
                if (aProb > 1e-8) entropy -= aProb * Math.log(aProb);
            }

            // 3. Calculate Variance of array elements for local dispersion scales
            const mean = row.reduce((a, b) => a + b, 0) / row.length;
            const variance = row.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / row.length;
            const stdDev = Math.sqrt(variance);

            // 4. Compute Microstructure Temperature (Tau = Variance * Entropy)
            const temperature = Math.max(entropy * stdDev, 1e-4);

            // 5. Compute Helmholtz Free Energy: F = U - TS
            const internalEnergy = totalVol;
            const freeEnergy = internalEnergy - (temperature * entropy);

            // 6. Compute Energy Barrier (Delta G): Half-spread + Kyle's Lambda instantaneous impact
            const halfSpread = (lowestAsk - highestBid) / 2.0;
            const topOfBookDepth = row[num] + row[3 * num];
            const energyBarrier = halfSpread + (0.05 * Math.sqrt(1.0 / (topOfBookDepth + 1e-6)));

            // 7. Compute Kramers' Escape Rate Formulation (State transition likelihood)
            const transitionProbability = Math.exp(-energyBarrier / (this.kB * temperature));

            return { 
                temperature, 
                freeEnergy, 
                energyBarrier, 
                transitionProbability 
            };
        });
    }
}
