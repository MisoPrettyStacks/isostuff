# 🌌 Latent Thermo-Trading Framework

[![GitHub License](https://shields.io)](LICENSE)
[![TensorFlow.js](https://shields.io)](https://tensorflow.org)
[![Node.js Version](https://shields.io)](https://nodejs.org)

An algorithmic framework that maps non-equilibrium thermodynamics to limit order book microstructures. The system couples a **Combinatory Synthesizer** (thermodynamic mapping), a **Latent Discovery Engine** (Physics-Regularized VAE via TensorFlow.js), and a **Multi-Agent Execution Layer** to trade across hidden market coordinates.

---

## 📐 Mathematical & Physical Isomorphisms

The framework bridges structural physics with financial mechanics by matching invariant mathematical properties:

| Thermodynamic Property | Financial Market Microstructure | Mathematical Formulation |
| :--- | :--- | :--- |
| **Microstates (\(x\))** | L2 Order Book Snapshot | Bid/Ask quantities across price levels \(q \in \mathbb{R}^{2K}\) |
| **Helmholtz Free Energy (\(F\))** | Available Effective Liquidity | \(F = U - TS \implies \Phi(x) = \text{Depth} - \tau \cdot H(p)\) |
| **Activation Barrier (\(\Delta G^\ddagger\))**| Order Execution Friction | \(\Delta G^\ddagger = \text{Half-Spread} + \kappa \cdot \text{Impact}(V)\) |
| **Temperature (\(\tau\))** | Microstructure Noise / Volatility | \(\tau = \text{Var}(\Delta p_{\Delta t}) / \mathbb{E}[\text{Volume}]\) |
| **Kramer's Escape Rate (\(k\))** | Fill & Boundary Crossing Probability | \(k \propto \exp\left(-\frac{\Delta G^\ddagger}{k_B \tau}\right)\) |

---

## 🏗 Architecture Blueprint

```text
 ┌───────────────────────────────────────────┐
 │        Level-2 Order Book Input           │
 └─────────────────────┬─────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
 ┌──────────────┐            ┌──────────────┐
 │Combinatory   │            │Latent        │
 │Synthesizer   │            │Discovery     │
 │(Thermodynamics)           │Engine (VAE)  │
 └───────┬──────┘            └───────┬──────┘
         │                           │
         │ [Temp, Energy, Barrier]   │ [Undiscovered Dimensions (Z)]
         └─────────────┬─────────────┘
                       ▼
 ┌───────────────────────────────────────────┐
 │        Multi-Agent Orchestrator           │
 │  ┌─────────────────────────────────────┐  │
 │  │ • Non-Equilibrium Arbitrageur       │  │
 │  │ • Entropy Momentum Agent            │  │
 │  │ • Barrier Liquidity Provider        │  │
 │  └──────────────────┬──────────────────┘  │
 └─────────────────────▼─────────────────────┘
                       │
                       ▼
        ┌─────────────────────────────┐
        │ Consolidated Portfolio Risk │
        └─────────────────────────────┘
```

### 1. Combinatory Synthesizer
Extracts instantaneous thermodynamic parameters directly from raw order book shapes. It models the book as an energy well where structural noise scales internal temperature ($\tau$), and the bid-ask spread operates as a physical potential barrier ($\Delta G^\ddagger$).

### 2. Latent Discovery Engine (VAE)
A Deep Variational Autoencoder built on **TensorFlow.js** that extracts low-dimensional latent variables ($Z \in \mathbb{R}^d$) called **Undiscovered Dimensions**. Regularization enforces strict consistency requirements, forcing the latent coordinate variance to align with micro-scale temperature configurations.

### 3. Multi-Agent Orchestration
Deploys specialized operational entities executing consensus weight allocation strategy optimized over specific coordinates of the discovered space:
* **Non-Equilibrium Arbitrageur:** Capitalizes on coordinate mean-reversion during intervals of elevated Kramers escape probabilities.
* **Entropy Momentum Agent:** Drives directional trend capture during compressed, high-free-energy regimes.
* **Barrier Liquidity Provider:** Harvests volatility when spreads stretch and structural escape transitions are blocked.

---

## 🚀 Getting Started

### Prerequisites
* [Node.js](https://nodejs.org) (Version 20.0.0 or higher required)
* NPM or Yarn package managers

### Installation

1. Clone this repository down to your local developer work environment:
   ```bash
   git clone https://github.com
   cd latent-thermo-trading
   ```

2. Populate all functional project dependencies:
   ```bash
   npm install
   ```

### Execution

Execute the core model pipeline verification simulator:
```bash
npm start
```

Run automated system validations:
```bash
npm test
```

---

## 🔧 Core Implementation Details

### Physics-Informed Loss Function
The system targets optimization across a combined loss matrix containing standard reconstruction accuracy alongside hard structural penalties:

$$\mathcal{L}_{\text{Total}} = \text{MSE}(x, \hat{x}) + \beta \cdot D_{\text{KL}}(q_\phi(z|x) \parallel p(z)) + \gamma \cdot \left\| \text{Var}(z_{\mu}) - \tau \right\|_2^2$$

Where $\text{Var}(z_{\mu})$ isolates the distribution spread of extracted features, forcing consistency directly matching incoming market engine temperatures ($\tau$).

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more details.
