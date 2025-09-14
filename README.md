# Chapter 231: VAE Factor Model for Financial Markets

This chapter explores **Variational Autoencoders (VAEs)** as a framework for discovering latent risk factors in financial markets. We show how the VAE latent space naturally maps to factor model concepts, providing a nonlinear generalization of classical approaches like PCA and Fama-French. The implementation uses Rust with live data from the Bybit exchange.

<p align="center">
<img src="https://i.imgur.com/Zx8KQPL.png" width="70%">
</p>

## Contents

1. [Introduction](#introduction)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Factor Models Meet VAEs](#factor-models-meet-vaes)
4. [Trading Applications](#trading-applications)
5. [Disentangled Factors](#disentangled-factors)
6. [Implementation Walkthrough](#implementation-walkthrough)
7. [Bybit Data Integration](#bybit-data-integration)
8. [Key Takeaways](#key-takeaways)

---

## Introduction

Traditional factor models in finance assume that the cross-section of asset returns is driven by a small number of latent risk factors. Principal Component Analysis (PCA) discovers these factors by finding orthogonal directions of maximum variance in the return covariance matrix, while the Fama-French model proposes pre-specified factors such as market beta, size, and value. Both approaches share a fundamental limitation: they assume linear relationships between factors and returns.

Financial markets, however, are rife with nonlinearities. Volatility clustering, regime shifts, asymmetric crash dynamics, and complex cross-asset dependencies all suggest that the true data-generating process is far from linear. A Variational Autoencoder offers an elegant solution: it learns a low-dimensional latent representation of the data through a nonlinear encoder-decoder architecture, while the probabilistic framework provides well-calibrated uncertainty estimates and the ability to generate new synthetic data points.

The key insight connecting VAEs to factor models is remarkably simple. In a linear factor model, observed returns **x** are generated as **x = Bf + e**, where **B** is a matrix of factor loadings, **f** is a vector of factor values, and **e** is idiosyncratic noise. In a VAE, the decoder network maps a latent vector **z** to reconstructed observations through a nonlinear function. If we interpret the latent variables **z** as risk factors and the decoder as a generalized (nonlinear) factor loading function, then the VAE becomes a universal nonlinear factor model.

This chapter develops the mathematics behind this correspondence, implements a VAE-based factor model in Rust, trains it on multi-asset cryptocurrency data from Bybit, and demonstrates practical trading applications including factor extraction, scenario generation, and risk decomposition.

## Mathematical Foundation

### The VAE Objective

A Variational Autoencoder learns a generative model of data **x** by introducing latent variables **z** and optimizing a tractable lower bound on the log-likelihood. The generative process assumes:

1. Sample latent factors from a prior: **z ~ p(z) = N(0, I)**
2. Generate observations from the decoder: **x ~ p_theta(x|z)**

Since the true posterior **p(z|x)** is intractable, we introduce an approximate posterior (the encoder) **q_phi(z|x)** and optimize the **Evidence Lower Bound (ELBO)**:

```
ELBO = E_{q_phi(z|x)}[log p_theta(x|z)] - KL[q_phi(z|x) || p(z)]
       \_______________________________/   \________________________/
          Reconstruction term                  KL divergence term
```

The **reconstruction term** encourages the decoder to accurately reconstruct the input data from the latent factors. In finance, this means the model must learn a latent representation from which the original return series can be recovered. The **KL divergence term** regularizes the latent space by pushing the approximate posterior toward the standard normal prior, preventing the model from memorizing the training data and ensuring a smooth, well-structured latent space.

### The Encoder: q_phi(z|x)

The encoder network maps observed data (e.g., multi-asset returns) to the parameters of a Gaussian distribution over latent factors:

```
q_phi(z|x) = N(z; mu_phi(x), diag(sigma^2_phi(x)))
```

where **mu_phi(x)** and **log(sigma^2_phi(x))** are the outputs of a neural network with parameters phi. Each input observation is mapped to a mean vector and a log-variance vector of the same dimensionality as the latent space. The diagonal covariance assumption keeps the model tractable while allowing each latent dimension to have a different variance.

### The Decoder: p_theta(x|z)

The decoder network maps latent factors back to the observation space:

```
p_theta(x|z) = N(x; f_theta(z), sigma^2_x * I)
```

where **f_theta(z)** is a nonlinear function parameterized by a neural network. In the context of factor models, this decoder plays the role of a generalized factor loading matrix, mapping abstract risk factors to concrete asset returns.

### The Reparameterization Trick

To enable gradient-based optimization through the stochastic sampling step, the VAE uses the reparameterization trick. Instead of sampling directly from **q_phi(z|x)**, we write:

```
z = mu + sigma * epsilon,    epsilon ~ N(0, I)
```

This reformulation moves the stochasticity to an input noise variable **epsilon** that does not depend on the model parameters, allowing gradients to flow through **mu** and **sigma** to the encoder weights.

### KL Divergence in Closed Form

For two Gaussian distributions, the KL divergence has a closed-form solution:

```
KL[q_phi(z|x) || p(z)] = -0.5 * sum_j (1 + log(sigma_j^2) - mu_j^2 - sigma_j^2)
```

where the sum runs over all latent dimensions j. This expression penalizes latent dimensions whose mean deviates from zero or whose variance deviates from one, encouraging the model to use each dimension efficiently.

## Factor Models Meet VAEs

### Latent z as Risk Factors

In a classical linear factor model:

```
CLASSICAL FACTOR MODEL:
===================================================================

  x = B * f + e

  x: N-dimensional return vector (N assets)
  B: N x K factor loading matrix
  f: K-dimensional factor vector
  e: N-dimensional idiosyncratic noise

  Key assumption: LINEARITY
===================================================================

VAE FACTOR MODEL:
===================================================================

  z ~ N(mu_phi(x), sigma^2_phi(x))     [encoder extracts factors]
  x_hat = f_theta(z)                     [decoder applies loadings]

  z: K-dimensional latent factor vector
  f_theta: nonlinear factor loading function
  sigma^2_phi: factor uncertainty estimates

  Key advantage: NONLINEARITY + UNCERTAINTY
===================================================================
```

The correspondence is direct: the latent vector **z** plays the role of factor values **f**, the decoder network plays the role of the loading matrix **B**, and the reconstruction error corresponds to idiosyncratic noise **e**. The crucial difference is that both the factor extraction (encoder) and the factor-to-return mapping (decoder) are nonlinear, allowing the model to capture complex market dynamics that linear models miss.

### Connecting to PCA

PCA is a special case of a VAE where both the encoder and decoder are linear and the KL regularization weight is set to zero. Specifically, if we restrict the encoder and decoder to be linear mappings (single-layer networks without activation functions), the VAE latent dimensions will converge to the principal components of the data. The KL regularization in the full VAE adds an important benefit: it rotates and scales the factors to be independent and unit-variance, providing a more interpretable decomposition.

### Connecting to Fama-French

The Fama-French model uses pre-specified factors (market, size, value, momentum, etc.) that are constructed from observable characteristics. The VAE discovers factors purely from data, without requiring any pre-specification. The VAE factors may capture the same phenomena as Fama-French factors (and empirically often do, especially the first few components), but they can also discover novel factors that are difficult for humans to pre-specify, such as complex nonlinear interactions between assets or regime-dependent risk exposures.

## Trading Applications

### Nonlinear Factor Discovery

The most direct application is discovering latent risk factors that drive the cross-section of returns. By training a VAE on multi-asset returns and examining the latent space, traders can identify:

- **Dominant market modes**: The latent dimensions that explain the most variance typically correspond to broad market movements, sector rotations, or risk-on/risk-off dynamics.
- **Nonlinear interactions**: Unlike PCA factors, VAE factors can capture asymmetric responses (e.g., assets that move differently in up vs. down markets).
- **Time-varying factor structure**: By examining how the encoder maps change over time, one can detect shifts in the factor structure that may signal regime changes.

### Generating Synthetic Market Scenarios

The generative nature of the VAE allows sampling from the latent space to create synthetic but realistic market scenarios:

```
SCENARIO GENERATION PIPELINE:
===================================================================

  1. Sample z ~ N(0, I) or z ~ N(mu_stressed, sigma_stressed)
  2. Decode: x_synthetic = f_theta(z)
  3. Use x_synthetic for:
     - Stress testing portfolios
     - Monte Carlo risk calculations
     - Training data augmentation
     - What-if analysis

  Advantage over historical simulation:
  - Can generate scenarios never seen in historical data
  - Scenarios respect the learned correlation structure
  - Can target specific factor regimes by conditioning z
===================================================================
```

### Risk Decomposition

The encoder provides a natural decomposition of portfolio risk into factor-driven and idiosyncratic components:

1. **Factor risk**: Encode portfolio returns to get latent factors and their uncertainties. The variance of the latent factors quantifies systematic risk.
2. **Idiosyncratic risk**: The reconstruction error (difference between actual and decoded returns) quantifies asset-specific risk not explained by the latent factors.
3. **Factor attribution**: By perturbing individual latent dimensions and observing the effect on decoded returns, one can attribute portfolio risk to specific latent factors.

## Disentangled Factors

A standard VAE may learn factors that are entangled — each latent dimension captures a mixture of multiple market phenomena. For trading applications, it is often desirable to have **disentangled factors** where each latent dimension corresponds to a single, interpretable market driver.

### Beta-VAE for Disentanglement

The beta-VAE modifies the ELBO by scaling the KL divergence term:

```
ELBO_beta = E[log p(x|z)] - beta * KL[q(z|x) || p(z)]
```

When **beta > 1**, the stronger regularization pressure forces the model to use each latent dimension more efficiently, encouraging disentanglement. Each dimension is pushed to capture a statistically independent source of variation. In financial markets, this might separate:

- **Dimension 1**: Broad market direction (analogous to market beta)
- **Dimension 2**: Relative value between large-cap and small-cap crypto
- **Dimension 3**: Volatility regime (high vs. low volatility)
- **Dimension 4**: Momentum vs. mean-reversion dynamics

The trade-off is that higher beta values reduce reconstruction quality. In practice, a beta between 1.0 and 4.0 usually provides a good balance between disentanglement and reconstruction accuracy for financial data.

### Measuring Disentanglement

To verify that the learned factors are truly disentangled, one can:

1. **Correlation analysis**: Compute the correlation matrix of latent activations across the training set. Disentangled factors should show near-zero off-diagonal correlations.
2. **Traversal analysis**: Sweep one latent dimension while holding others fixed and observe which aspects of the decoded output change. A disentangled factor should affect a single, coherent aspect of the market.
3. **Factor Sharpe ratios**: Construct long-short portfolios based on individual latent factor exposures. Disentangled factors should produce return streams with distinct risk/return profiles.

## Implementation Walkthrough

The Rust implementation in this chapter provides a complete VAE factor model with the following components:

### Core Architecture

```rust
// The VAE consists of encoder and decoder networks
// Encoder: input_dim -> hidden_dim -> (mu, log_var) of latent_dim
// Decoder: latent_dim -> hidden_dim -> input_dim

pub struct VaeFactor {
    // Encoder weights
    encoder_w1: Array2<f64>,  // input -> hidden
    encoder_b1: Array1<f64>,
    encoder_w_mu: Array2<f64>,  // hidden -> mu
    encoder_b_mu: Array1<f64>,
    encoder_w_logvar: Array2<f64>,  // hidden -> log_var
    encoder_b_logvar: Array1<f64>,

    // Decoder weights
    decoder_w1: Array2<f64>,  // latent -> hidden
    decoder_b1: Array1<f64>,
    decoder_w2: Array2<f64>,  // hidden -> output
    decoder_b2: Array1<f64>,

    pub config: VaeConfig,
}
```

### Training Loop

The training loop performs stochastic gradient descent on the negative ELBO. Each iteration:

1. Forward pass through the encoder to get mu and log_var
2. Sample z using the reparameterization trick
3. Forward pass through the decoder to reconstruct the input
4. Compute the ELBO loss (reconstruction MSE + beta * KL divergence)
5. Backpropagate gradients through the entire network
6. Update weights using gradient descent

### Factor Extraction and Analysis

After training, the model provides:

- **`encode()`**: Maps market observations to latent factor values with uncertainty estimates
- **`decode()`**: Maps factor values to predicted market returns
- **`generate_scenarios()`**: Samples from the prior to create synthetic market scenarios
- **`factor_loadings()`**: Computes the sensitivity of each asset to each latent factor via finite differences

## Bybit Data Integration

The implementation fetches real market data from the Bybit exchange via its public REST API:

```rust
// Fetch kline data for multiple assets
let symbols = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT"];
let loader = BybitDataLoader::new();

for symbol in &symbols {
    let candles = loader.fetch_klines(symbol, "1h", 500).await?;
    // Process candles into return series...
}
```

The Bybit integration allows the model to be trained on live cryptocurrency market data across multiple assets, capturing real cross-asset dynamics and correlations. The multi-asset return matrix serves as the input to the VAE, where each row is a time step and each column is an asset's return.

### Data Preprocessing

Returns are computed as log-returns and standardized (zero mean, unit variance) before feeding into the VAE. This normalization is critical because:

1. It puts all assets on a comparable scale regardless of their price levels
2. It centers the data around zero, matching the zero-mean prior on the latent space
3. It improves gradient flow during training by avoiding very large or very small activations

## Key Takeaways

1. **VAEs generalize linear factor models**: The latent space of a VAE is a nonlinear analog of the factor space in PCA or Fama-French models. The encoder extracts factors, and the decoder applies (nonlinear) factor loadings.

2. **The ELBO provides a principled objective**: The reconstruction term ensures the factors are informative, while the KL divergence term ensures the factor space is regular and well-structured.

3. **Reparameterization enables end-to-end learning**: The reparameterization trick allows gradient-based optimization through the stochastic sampling step, making the entire model differentiable.

4. **Disentanglement improves interpretability**: Using beta-VAE with beta > 1 encourages each latent dimension to capture a distinct market phenomenon, making the factors more interpretable and useful for risk management.

5. **Scenario generation goes beyond historical data**: By sampling from the latent space, traders can generate synthetic but realistic market scenarios that may never have occurred historically, enabling more robust stress testing and risk assessment.

6. **Factor loadings reveal asset sensitivities**: The decoder Jacobian (approximated via finite differences) reveals how each asset responds to each latent factor, providing a nonlinear analog of the factor loading matrix.

7. **Cryptocurrency markets are ideal testing grounds**: The high volatility, 24/7 trading, and complex cross-asset dynamics of crypto markets provide rich structure for VAE factor models to discover.

8. **Rust provides production-grade performance**: The Rust implementation offers memory safety without garbage collection, making it suitable for real-time factor monitoring and risk calculations in production trading systems.

## Resources

- Kingma & Welling (2013). *Auto-Encoding Variational Bayes*. arXiv:1312.6114
- Higgins et al. (2017). *beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework*. ICLR 2017
- Gu, Kelly & Xiu (2020). *Autoencoder Asset Pricing Models*. Journal of Econometrics
- Chen et al. (2020). *Deep Learning in Asset Pricing*. Management Science
- Bybit API Documentation: https://bybit-exchange.github.io/docs/v5/intro
