/**
 * Utility tool to simulate valid Level-2 limit order book tracking records.
 */
export class SyntheticOrderBookGenerator {
    /**
     * @param {number} numLevels - The total number of price-depth rows per side.
     */
    constructor(numLevels = 10) {
        this.numLevels = numLevels;
        this.inputDim = 4 * numLevels; // [bid_px, bid_sz, ask_px, ask_sz]
    }

    /**
     * Generates a batch of synthetic order book frames with realistic structural characteristics.
     * @param {number} batchSize - Number of snapshots to create.
     * @returns {number[][]} Matrix block matching the shape constraints of the processing layers.
     */
    generateBatch(batchSize = 4) {
        const batch = [];
        
        for (let i = 0; i < batchSize; i++) {
            const row = Array.from({ length: this.inputDim }, () => Math.random());
            
            // Slice out independent pricing tracks
            const rawBids = row.slice(0, this.numLevels);
            const rawAsks = row.slice(2 * this.numLevels, 3 * this.numLevels);
            
            // Enforce realistic ordering properties (Bids descending, Asks ascending)
            const sortedBids = rawBids.sort((a, b) => b - a);
            const sortedAsks = rawAsks.sort((a, b) => a - b);
            
            // Adjust ask prices to guarantee a realistic spread gap above bids
            const bidCeiling = sortedBids[0];
            const correctedAsks = sortedAsks.map(px => bidCeiling + 0.01 + px * 0.1);
            
            // Splice back the structural price records
            row.splice(0, this.numLevels, ...sortedBids);
            row.splice(2 * this.numLevels, this.numLevels, ...correctedAsks);
            
            batch.push(row);
        }
        
        return batch;
    }
}
