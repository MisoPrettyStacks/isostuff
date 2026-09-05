/**
 * Abstract Base Class for autonomous agents operating within latent coordinates.
 */
export class LatentDimensionAgent {
    /**
     * @param {string} name - Identifying name of the execution agent.
     * @param {number} focusDim - The specific index of the hidden coordinate Z handled by this agent.
     */
    constructor(name, focusDim) {
        if (this.constructor === LatentDimensionAgent) {
            throw new TypeError("Cannot construct LatentDimensionAgent instances directly.");
        }
        this.name = name;
        this.focusDim = focusDim;
    }

    /**
     * Operational logic boundary to execute a trade action based on coordinate mappings.
     * @param {number[]} z - Latent dimension array for the single sample.
     * @param {Object} thermo - Dictionary containing computed thermodynamic variables.
     * @returns {Object} Target allocation map: { agent: string, action: number, conviction: number }
     */
    act(z, thermo) {
        throw new Error("Method 'act()' must be implemented by subclasses.");
    }
}
