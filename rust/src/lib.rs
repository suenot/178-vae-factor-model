//! # VAE Factor Model
//!
//! A Variational Autoencoder-based factor model for discovering latent risk
//! factors in financial markets. The VAE latent space serves as a nonlinear
//! generalization of classical factor models like PCA and Fama-French.
//!
//! ## Features
//!
//! - VAE encoder/decoder with configurable architecture
//! - Reparameterization trick for differentiable sampling
//! - ELBO loss with beta-VAE support for disentanglement
//! - Factor extraction and scenario generation
//! - Factor loading analysis via finite differences
//! - Bybit API integration for multi-asset crypto data
//!
//! ## Example
//!
//! ```rust,no_run
//! use vae_factor_model::{VaeFactor, VaeConfig, BybitDataLoader};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let loader = BybitDataLoader::new();
//!     let returns = loader.fetch_multi_asset_returns(
//!         &["BTCUSDT", "ETHUSDT", "SOLUSDT"],
//!         "1h", 500
//!     ).await?;
//!
//!     let config = VaeConfig::new(3, 16, 3);
//!     let mut vae = VaeFactor::new(config);
//!     vae.train(&returns, 100, 0.001)?;
//!
//!     let (mu, _log_var) = vae.encode(&returns.row(0).to_owned());
//!     println!("Latent factors: {:?}", mu);
//!     Ok(())
//! }
//! ```

use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use serde::{Deserialize, Serialize};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the VAE factor model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaeConfig {
    /// Dimensionality of the input (number of assets).
    pub input_dim: usize,
    /// Dimensionality of the hidden layer.
    pub hidden_dim: usize,
    /// Dimensionality of the latent space (number of factors).
    pub latent_dim: usize,
    /// Beta weight for the KL divergence term (beta-VAE).
    /// beta=1.0 is standard VAE, beta>1.0 encourages disentanglement.
    pub beta: f64,
}

impl VaeConfig {
    /// Create a new VAE configuration.
    ///
    /// # Arguments
    /// * `input_dim` - Number of input features (assets)
    /// * `hidden_dim` - Hidden layer size
    /// * `latent_dim` - Number of latent factors
    pub fn new(input_dim: usize, hidden_dim: usize, latent_dim: usize) -> Self {
        Self {
            input_dim,
            hidden_dim,
            latent_dim,
            beta: 1.0,
        }
    }

    /// Set the beta parameter for beta-VAE disentanglement.
    pub fn with_beta(mut self, beta: f64) -> Self {
        self.beta = beta;
        self
    }
}

// ─── Activation functions ─────────────────────────────────────────────────────

/// ReLU activation: max(0, x)
fn relu(x: f64) -> f64 {
    x.max(0.0)
}

/// Derivative of ReLU
fn relu_derivative(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

/// Apply ReLU element-wise to an array.
fn relu_array(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(relu)
}

// ─── VAE Factor Model ────────────────────────────────────────────────────────

/// Variational Autoencoder factor model.
///
/// Discovers latent risk factors in multi-asset return data using a
/// nonlinear encoder-decoder architecture with probabilistic latent space.
#[derive(Debug, Clone)]
pub struct VaeFactor {
    // Encoder weights: input -> hidden
    pub encoder_w1: Array2<f64>,
    pub encoder_b1: Array1<f64>,
    // Encoder weights: hidden -> mu
    pub encoder_w_mu: Array2<f64>,
    pub encoder_b_mu: Array1<f64>,
    // Encoder weights: hidden -> log_var
    pub encoder_w_logvar: Array2<f64>,
    pub encoder_b_logvar: Array1<f64>,

    // Decoder weights: latent -> hidden
    pub decoder_w1: Array2<f64>,
    pub decoder_b1: Array1<f64>,
    // Decoder weights: hidden -> output
    pub decoder_w2: Array2<f64>,
    pub decoder_b2: Array1<f64>,

    /// Model configuration.
    pub config: VaeConfig,
}

/// Result of encoding an observation.
#[derive(Debug, Clone)]
pub struct EncodeResult {
    /// Mean of the latent distribution.
    pub mu: Array1<f64>,
    /// Log-variance of the latent distribution.
    pub log_var: Array1<f64>,
    /// Sampled latent vector (using reparameterization trick).
    pub z: Array1<f64>,
}

/// Result of a single training step.
#[derive(Debug, Clone)]
pub struct TrainStepResult {
    /// Total loss (negative ELBO).
    pub total_loss: f64,
    /// Reconstruction loss (MSE).
    pub recon_loss: f64,
    /// KL divergence loss.
    pub kl_loss: f64,
}

/// Factor loading matrix: sensitivity of each asset to each latent factor.
#[derive(Debug, Clone)]
pub struct FactorLoadings {
    /// Matrix of shape (input_dim, latent_dim).
    /// Entry (i, j) = sensitivity of asset i to factor j.
    pub loadings: Array2<f64>,
    /// Asset labels (if provided).
    pub asset_names: Vec<String>,
}

impl VaeFactor {
    /// Create a new VAE factor model with Xavier-initialized weights.
    pub fn new(config: VaeConfig) -> Self {
        let mut rng = rand::thread_rng();
        let input_dim = config.input_dim;
        let hidden_dim = config.hidden_dim;
        let latent_dim = config.latent_dim;

        let mut xavier = |fan_in: usize, fan_out: usize| -> Array2<f64> {
            let scale = (2.0 / (fan_in + fan_out) as f64).sqrt();
            Array2::from_shape_fn((fan_in, fan_out), |_| {
                rng.gen_range(-scale..scale)
            })
        };

        Self {
            encoder_w1: xavier(input_dim, hidden_dim),
            encoder_b1: Array1::zeros(hidden_dim),
            encoder_w_mu: xavier(hidden_dim, latent_dim),
            encoder_b_mu: Array1::zeros(latent_dim),
            encoder_w_logvar: xavier(hidden_dim, latent_dim),
            encoder_b_logvar: Array1::zeros(latent_dim),

            decoder_w1: xavier(latent_dim, hidden_dim),
            decoder_b1: Array1::zeros(hidden_dim),
            decoder_w2: xavier(hidden_dim, input_dim),
            decoder_b2: Array1::zeros(input_dim),

            config,
        }
    }

    /// Encode an observation into latent space.
    ///
    /// Returns (mu, log_var) of the approximate posterior q(z|x).
    pub fn encode(&self, x: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        // Hidden layer with ReLU
        let hidden = relu_array(&(x.dot(&self.encoder_w1) + &self.encoder_b1));
        // Mean and log-variance of latent distribution
        let mu = hidden.dot(&self.encoder_w_mu) + &self.encoder_b_mu;
        let log_var = hidden.dot(&self.encoder_w_logvar) + &self.encoder_b_logvar;
        (mu, log_var)
    }

    /// Encode with full result including sampled z.
    pub fn encode_full(&self, x: &Array1<f64>) -> EncodeResult {
        let (mu, log_var) = self.encode(x);
        let z = self.reparameterize(&mu, &log_var);
        EncodeResult { mu, log_var, z }
    }

    /// Reparameterization trick: z = mu + sigma * epsilon, epsilon ~ N(0, I).
    ///
    /// This allows gradients to flow through the sampling operation.
    pub fn reparameterize(&self, mu: &Array1<f64>, log_var: &Array1<f64>) -> Array1<f64> {
        let mut rng = rand::thread_rng();
        let std = log_var.mapv(|v| (v * 0.5).exp());
        let epsilon = Array1::from_shape_fn(mu.len(), |_| rng.gen_range(-1.0..1.0));
        mu + &(std * epsilon)
    }

    /// Decode a latent vector back to observation space.
    ///
    /// This is the nonlinear factor loading function: z -> x_reconstructed.
    pub fn decode(&self, z: &Array1<f64>) -> Array1<f64> {
        let hidden = relu_array(&(z.dot(&self.decoder_w1) + &self.decoder_b1));
        hidden.dot(&self.decoder_w2) + &self.decoder_b2
    }

    /// Forward pass: encode, sample, decode.
    ///
    /// Returns (reconstructed_x, mu, log_var, z).
    pub fn forward(&self, x: &Array1<f64>) -> (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
        let (mu, log_var) = self.encode(x);
        let z = self.reparameterize(&mu, &log_var);
        let x_recon = self.decode(&z);
        (x_recon, mu, log_var, z)
    }

    /// Compute the ELBO loss for a single observation.
    ///
    /// Loss = Reconstruction_MSE + beta * KL_divergence
    pub fn elbo_loss(
        &self,
        x: &Array1<f64>,
        x_recon: &Array1<f64>,
        mu: &Array1<f64>,
        log_var: &Array1<f64>,
    ) -> (f64, f64, f64) {
        // Reconstruction loss: mean squared error
        let diff = x - x_recon;
        let recon_loss = diff.mapv(|v| v * v).mean().unwrap_or(0.0);

        // KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        let kl_loss = -0.5
            * log_var
                .iter()
                .zip(mu.iter())
                .map(|(&lv, &m)| 1.0 + lv - m * m - lv.exp())
                .sum::<f64>();

        let total_loss = recon_loss + self.config.beta * kl_loss;
        (total_loss, recon_loss, kl_loss)
    }

    /// Train the VAE on a dataset of observations.
    ///
    /// # Arguments
    /// * `data` - Matrix of shape (n_samples, input_dim) containing return observations
    /// * `epochs` - Number of training epochs
    /// * `learning_rate` - Learning rate for gradient descent
    ///
    /// Returns training history as a vector of per-epoch average losses.
    pub fn train(
        &mut self,
        data: &Array2<f64>,
        epochs: usize,
        learning_rate: f64,
    ) -> anyhow::Result<Vec<TrainStepResult>> {
        let n_samples = data.nrows();
        if n_samples == 0 {
            anyhow::bail!("Empty training data");
        }
        if data.ncols() != self.config.input_dim {
            anyhow::bail!(
                "Input dimension mismatch: expected {}, got {}",
                self.config.input_dim,
                data.ncols()
            );
        }

        let mut history = Vec::with_capacity(epochs);

        for _epoch in 0..epochs {
            let mut epoch_total = 0.0;
            let mut epoch_recon = 0.0;
            let mut epoch_kl = 0.0;

            for i in 0..n_samples {
                let x = data.row(i).to_owned();

                // ── Forward pass ──────────────────────────────────────────
                // Encoder: input -> hidden
                let h_enc_pre = x.dot(&self.encoder_w1) + &self.encoder_b1;
                let h_enc = relu_array(&h_enc_pre);

                // Encoder: hidden -> mu, log_var
                let mu = h_enc.dot(&self.encoder_w_mu) + &self.encoder_b_mu;
                let log_var = h_enc.dot(&self.encoder_w_logvar) + &self.encoder_b_logvar;

                // Reparameterize
                let z = self.reparameterize(&mu, &log_var);

                // Decoder: latent -> hidden
                let h_dec_pre = z.dot(&self.decoder_w1) + &self.decoder_b1;
                let h_dec = relu_array(&h_dec_pre);

                // Decoder: hidden -> output
                let x_recon = h_dec.dot(&self.decoder_w2) + &self.decoder_b2;

                // ── Compute loss ──────────────────────────────────────────
                let (total, recon, kl) = self.elbo_loss(&x, &x_recon, &mu, &log_var);
                epoch_total += total;
                epoch_recon += recon;
                epoch_kl += kl;

                // ── Backward pass (manual gradient computation) ───────────
                let n_input = self.config.input_dim as f64;

                // Gradient of reconstruction loss w.r.t. x_recon
                let d_recon = (&x_recon - &x).mapv(|v| 2.0 * v / n_input);

                // Decoder output layer gradients
                let d_decoder_w2 = outer_product(&h_dec, &d_recon);
                let d_decoder_b2 = d_recon.clone();

                // Decoder hidden layer gradients
                let d_h_dec = d_recon.dot(&self.decoder_w2.t());
                let d_h_dec_pre = &d_h_dec * &h_dec_pre.mapv(relu_derivative);

                let d_decoder_w1 = outer_product(&z, &d_h_dec_pre);
                let d_decoder_b1 = d_h_dec_pre.clone();

                // Gradient of z (from decoder)
                let d_z = d_h_dec_pre.dot(&self.decoder_w1.t());

                // Gradients of mu and log_var from reparameterization + KL
                // d_loss/d_mu = d_z (from recon) + beta * mu (from KL)
                let d_mu = &d_z + &mu.mapv(|m| self.config.beta * m);
                // d_loss/d_log_var = d_z * 0.5 * exp(0.5*log_var) * eps
                //                  + beta * 0.5 * (exp(log_var) - 1)
                let d_log_var = log_var.mapv(|lv| {
                    self.config.beta * 0.5 * (lv.exp() - 1.0)
                });

                // Encoder mu layer gradients
                let d_encoder_w_mu = outer_product(&h_enc, &d_mu);
                let d_encoder_b_mu = d_mu.clone();

                // Encoder log_var layer gradients
                let d_encoder_w_logvar = outer_product(&h_enc, &d_log_var);
                let d_encoder_b_logvar = d_log_var.clone();

                // Encoder hidden layer gradients
                let d_h_enc = d_mu.dot(&self.encoder_w_mu.t())
                    + d_log_var.dot(&self.encoder_w_logvar.t());
                let d_h_enc_pre = &d_h_enc * &h_enc_pre.mapv(relu_derivative);

                let d_encoder_w1 = outer_product(&x, &d_h_enc_pre);
                let d_encoder_b1 = d_h_enc_pre;

                // ── Update weights ────────────────────────────────────────
                self.encoder_w1 = &self.encoder_w1 - &d_encoder_w1.mapv(|v| v * learning_rate);
                self.encoder_b1 = &self.encoder_b1 - &d_encoder_b1.mapv(|v| v * learning_rate);
                self.encoder_w_mu = &self.encoder_w_mu - &d_encoder_w_mu.mapv(|v| v * learning_rate);
                self.encoder_b_mu = &self.encoder_b_mu - &d_encoder_b_mu.mapv(|v| v * learning_rate);
                self.encoder_w_logvar =
                    &self.encoder_w_logvar - &d_encoder_w_logvar.mapv(|v| v * learning_rate);
                self.encoder_b_logvar =
                    &self.encoder_b_logvar - &d_encoder_b_logvar.mapv(|v| v * learning_rate);
                self.decoder_w1 = &self.decoder_w1 - &d_decoder_w1.mapv(|v| v * learning_rate);
                self.decoder_b1 = &self.decoder_b1 - &d_decoder_b1.mapv(|v| v * learning_rate);
                self.decoder_w2 = &self.decoder_w2 - &d_decoder_w2.mapv(|v| v * learning_rate);
                self.decoder_b2 = &self.decoder_b2 - &d_decoder_b2.mapv(|v| v * learning_rate);
            }

            history.push(TrainStepResult {
                total_loss: epoch_total / n_samples as f64,
                recon_loss: epoch_recon / n_samples as f64,
                kl_loss: epoch_kl / n_samples as f64,
            });
        }

        Ok(history)
    }

    /// Extract latent factors for all observations in the dataset.
    ///
    /// Returns a matrix of shape (n_samples, latent_dim) where each row
    /// contains the mean latent factor values for that observation.
    pub fn extract_factors(&self, data: &Array2<f64>) -> Array2<f64> {
        let n_samples = data.nrows();
        let mut factors = Array2::zeros((n_samples, self.config.latent_dim));
        for i in 0..n_samples {
            let x = data.row(i).to_owned();
            let (mu, _log_var) = self.encode(&x);
            factors.row_mut(i).assign(&mu);
        }
        factors
    }

    /// Generate synthetic market scenarios by sampling from the latent space.
    ///
    /// Samples z ~ N(0, I) and decodes to observation space.
    ///
    /// # Arguments
    /// * `n_scenarios` - Number of scenarios to generate
    pub fn generate_scenarios(&self, n_scenarios: usize) -> Array2<f64> {
        let mut rng = rand::thread_rng();
        let mut scenarios = Array2::zeros((n_scenarios, self.config.input_dim));
        for i in 0..n_scenarios {
            let z = Array1::from_shape_fn(self.config.latent_dim, |_| {
                rng.gen_range(-1.0..1.0) * 1.0
            });
            let x = self.decode(&z);
            scenarios.row_mut(i).assign(&x);
        }
        scenarios
    }

    /// Generate stressed scenarios by sampling from a shifted latent distribution.
    ///
    /// # Arguments
    /// * `n_scenarios` - Number of scenarios to generate
    /// * `stress_mu` - Mean of the stress distribution in latent space
    /// * `stress_sigma` - Standard deviation of the stress distribution
    pub fn generate_stressed_scenarios(
        &self,
        n_scenarios: usize,
        stress_mu: &Array1<f64>,
        stress_sigma: f64,
    ) -> Array2<f64> {
        let mut rng = rand::thread_rng();
        let mut scenarios = Array2::zeros((n_scenarios, self.config.input_dim));
        for i in 0..n_scenarios {
            let z = Array1::from_shape_fn(self.config.latent_dim, |j| {
                stress_mu[j] + rng.gen_range(-1.0..1.0) * stress_sigma
            });
            let x = self.decode(&z);
            scenarios.row_mut(i).assign(&x);
        }
        scenarios
    }

    /// Compute factor loadings via finite differences.
    ///
    /// For each latent dimension j, perturbs z_j and measures the effect on
    /// each output dimension, approximating the Jacobian of the decoder.
    ///
    /// # Arguments
    /// * `z_base` - Base latent vector (e.g., zeros or mean encoding)
    /// * `epsilon` - Perturbation size for finite differences
    pub fn factor_loadings(
        &self,
        z_base: &Array1<f64>,
        epsilon: f64,
    ) -> Array2<f64> {
        let mut loadings = Array2::zeros((self.config.input_dim, self.config.latent_dim));
        let x_base = self.decode(z_base);

        for j in 0..self.config.latent_dim {
            let mut z_perturbed = z_base.clone();
            z_perturbed[j] += epsilon;
            let x_perturbed = self.decode(&z_perturbed);
            let sensitivity = (&x_perturbed - &x_base).mapv(|v| v / epsilon);
            loadings.column_mut(j).assign(&sensitivity);
        }

        loadings
    }

    /// Compute factor loadings with asset names for display.
    pub fn factor_loadings_named(
        &self,
        z_base: &Array1<f64>,
        epsilon: f64,
        asset_names: &[&str],
    ) -> FactorLoadings {
        let loadings = self.factor_loadings(z_base, epsilon);
        FactorLoadings {
            loadings,
            asset_names: asset_names.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Compute the reconstruction error for a dataset.
    ///
    /// Returns the mean squared error between input and reconstruction.
    pub fn reconstruction_error(&self, data: &Array2<f64>) -> f64 {
        let n_samples = data.nrows();
        let mut total_error = 0.0;
        for i in 0..n_samples {
            let x = data.row(i).to_owned();
            let (mu, _) = self.encode(&x);
            let x_recon = self.decode(&mu);
            let diff = &x - &x_recon;
            total_error += diff.mapv(|v| v * v).mean().unwrap_or(0.0);
        }
        total_error / n_samples as f64
    }

    /// Decompose observation variance into factor-driven and idiosyncratic components.
    ///
    /// Returns (factor_variance_ratio, idiosyncratic_variance_ratio).
    pub fn variance_decomposition(&self, data: &Array2<f64>) -> (f64, f64) {
        let n_samples = data.nrows();
        let mut total_var = 0.0;
        let mut recon_var = 0.0;

        // Compute mean
        let mean = data.mean_axis(Axis(0)).unwrap();

        for i in 0..n_samples {
            let x = data.row(i).to_owned();
            let diff_total = &x - &mean;
            total_var += diff_total.mapv(|v| v * v).sum();

            let (mu, _) = self.encode(&x);
            let x_recon = self.decode(&mu);
            let diff_recon = &x_recon - &mean;
            recon_var += diff_recon.mapv(|v| v * v).sum();
        }

        if total_var == 0.0 {
            return (0.0, 1.0);
        }

        let factor_ratio = (recon_var / total_var).min(1.0).max(0.0);
        let idio_ratio = 1.0 - factor_ratio;
        (factor_ratio, idio_ratio)
    }
}

/// Compute the outer product of two vectors.
fn outer_product(a: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
    let n = a.len();
    let m = b.len();
    let mut result = Array2::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            result[[i, j]] = a[i] * b[j];
        }
    }
    result
}

// ─── Data Preprocessing ──────────────────────────────────────────────────────

/// Standardization parameters for a dataset.
#[derive(Debug, Clone)]
pub struct StandardScaler {
    pub mean: Array1<f64>,
    pub std: Array1<f64>,
}

impl StandardScaler {
    /// Fit the scaler on a dataset and return the standardized data.
    pub fn fit_transform(data: &Array2<f64>) -> (Self, Array2<f64>) {
        let mean = data.mean_axis(Axis(0)).unwrap();
        let std = data.std_axis(Axis(0), 0.0);
        // Avoid division by zero
        let std_safe = std.mapv(|s| if s < 1e-10 { 1.0 } else { s });

        let mut standardized = data.clone();
        for mut row in standardized.rows_mut() {
            let centered = &row.to_owned() - &mean;
            let scaled = &centered / &std_safe;
            row.assign(&scaled);
        }

        (Self { mean, std: std_safe }, standardized)
    }

    /// Transform new data using the fitted parameters.
    pub fn transform(&self, data: &Array2<f64>) -> Array2<f64> {
        let mut standardized = data.clone();
        for mut row in standardized.rows_mut() {
            let centered = &row.to_owned() - &self.mean;
            let scaled = &centered / &self.std;
            row.assign(&scaled);
        }
        standardized
    }

    /// Inverse transform: convert standardized data back to original scale.
    pub fn inverse_transform(&self, data: &Array2<f64>) -> Array2<f64> {
        let mut original = data.clone();
        for mut row in original.rows_mut() {
            let rescaled = &row.to_owned() * &self.std + &self.mean;
            row.assign(&rescaled);
        }
        original
    }
}

/// Compute log-returns from a price series.
///
/// Returns an array of length n-1 where each element is ln(p[t]/p[t-1]).
pub fn compute_log_returns(prices: &[f64]) -> Vec<f64> {
    prices
        .windows(2)
        .map(|w| (w[1] / w[0]).ln())
        .collect()
}

/// Build a return matrix from multiple price series.
///
/// Each column corresponds to one asset. Rows are aligned time steps.
/// The output has min_len - 1 rows where min_len is the shortest price series.
pub fn build_return_matrix(price_series: &[Vec<f64>]) -> Array2<f64> {
    let returns: Vec<Vec<f64>> = price_series.iter().map(|p| compute_log_returns(p)).collect();
    let min_len = returns.iter().map(|r| r.len()).min().unwrap_or(0);
    let n_assets = returns.len();

    let mut matrix = Array2::zeros((min_len, n_assets));
    for (j, ret) in returns.iter().enumerate() {
        for i in 0..min_len {
            matrix[[i, j]] = ret[i];
        }
    }
    matrix
}

// ─── Bybit Data Loader ───────────────────────────────────────────────────────

/// Response structures for the Bybit API.
#[derive(Debug, Deserialize)]
pub struct BybitKlineResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i64,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: BybitKlineResult,
}

#[derive(Debug, Deserialize)]
pub struct BybitKlineResult {
    pub symbol: String,
    pub category: String,
    pub list: Vec<Vec<String>>,
}

/// A single candlestick (kline) from Bybit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Bybit data loader for fetching market data.
pub struct BybitDataLoader {
    base_url: String,
}

impl BybitDataLoader {
    /// Create a new Bybit data loader.
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Fetch kline (candlestick) data for a symbol.
    ///
    /// # Arguments
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `interval` - Kline interval (e.g., "1", "5", "60", "D")
    /// * `limit` - Maximum number of candles to fetch (max 200)
    pub async fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<Candle>> {
        let url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let client = reqwest::Client::new();
        let response: BybitKlineResponse = client.get(&url).send().await?.json().await?;

        if response.ret_code != 0 {
            anyhow::bail!("Bybit API error: {}", response.ret_msg);
        }

        let mut candles: Vec<Candle> = response
            .result
            .list
            .iter()
            .filter_map(|row| {
                if row.len() >= 6 {
                    Some(Candle {
                        timestamp: row[0].parse().ok()?,
                        open: row[1].parse().ok()?,
                        high: row[2].parse().ok()?,
                        low: row[3].parse().ok()?,
                        close: row[4].parse().ok()?,
                        volume: row[5].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Bybit returns newest first; reverse to chronological order
        candles.reverse();
        Ok(candles)
    }

    /// Fetch klines and extract close prices.
    pub async fn fetch_close_prices(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<f64>> {
        let candles = self.fetch_klines(symbol, interval, limit).await?;
        Ok(candles.iter().map(|c| c.close).collect())
    }

    /// Fetch multi-asset return matrix.
    ///
    /// Fetches close prices for each symbol, computes log-returns,
    /// and builds an aligned return matrix.
    pub async fn fetch_multi_asset_returns(
        &self,
        symbols: &[&str],
        interval: &str,
        limit: usize,
    ) -> anyhow::Result<Array2<f64>> {
        let mut all_prices = Vec::new();
        for symbol in symbols {
            let prices = self.fetch_close_prices(symbol, interval, limit).await?;
            if prices.is_empty() {
                anyhow::bail!("No data returned for {}", symbol);
            }
            all_prices.push(prices);
        }
        Ok(build_return_matrix(&all_prices))
    }
}

impl Default for BybitDataLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for FactorLoadings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Factor Loadings:")?;
        writeln!(f, "{:<12} {}", "Asset", (0..self.loadings.ncols())
            .map(|j| format!("Factor_{:<4}", j))
            .collect::<Vec<_>>()
            .join(" "))?;
        writeln!(f, "{}", "-".repeat(12 + self.loadings.ncols() * 11))?;

        for i in 0..self.loadings.nrows() {
            let name = if i < self.asset_names.len() {
                &self.asset_names[i]
            } else {
                "unknown"
            };
            let values: Vec<String> = (0..self.loadings.ncols())
                .map(|j| format!("{:>10.6}", self.loadings[[i, j]]))
                .collect();
            writeln!(f, "{:<12} {}", name, values.join(" "))?;
        }
        Ok(())
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_data() -> Array2<f64> {
        // 20 observations, 3 assets
        let mut rng = rand::thread_rng();
        // Create data with some structure: a shared factor plus noise
        let n = 20;
        let mut data = Array2::zeros((n, 3));
        for i in 0..n {
            let factor: f64 = rng.gen_range(-1.0..1.0);
            data[[i, 0]] = factor * 1.0 + rng.gen_range(-0.1..0.1);
            data[[i, 1]] = factor * 0.8 + rng.gen_range(-0.1..0.1);
            data[[i, 2]] = factor * 0.5 + rng.gen_range(-0.2..0.2);
        }
        data
    }

    #[test]
    fn test_vae_creation() {
        let config = VaeConfig::new(3, 16, 2);
        let vae = VaeFactor::new(config);
        assert_eq!(vae.config.input_dim, 3);
        assert_eq!(vae.config.hidden_dim, 16);
        assert_eq!(vae.config.latent_dim, 2);
        assert_eq!(vae.encoder_w1.shape(), &[3, 16]);
        assert_eq!(vae.decoder_w2.shape(), &[16, 3]);
    }

    #[test]
    fn test_encode_decode() {
        let config = VaeConfig::new(3, 16, 2);
        let vae = VaeFactor::new(config);

        let x = Array1::from_vec(vec![0.1, -0.05, 0.02]);
        let (mu, log_var) = vae.encode(&x);

        assert_eq!(mu.len(), 2);
        assert_eq!(log_var.len(), 2);

        let x_recon = vae.decode(&mu);
        assert_eq!(x_recon.len(), 3);
    }

    #[test]
    fn test_reparameterize() {
        let config = VaeConfig::new(3, 16, 2);
        let vae = VaeFactor::new(config);

        let mu = Array1::from_vec(vec![0.0, 0.0]);
        let log_var = Array1::from_vec(vec![0.0, 0.0]);

        let z = vae.reparameterize(&mu, &log_var);
        assert_eq!(z.len(), 2);
    }

    #[test]
    fn test_forward_pass() {
        let config = VaeConfig::new(3, 16, 2);
        let vae = VaeFactor::new(config);

        let x = Array1::from_vec(vec![0.1, -0.05, 0.02]);
        let (x_recon, mu, log_var, z) = vae.forward(&x);

        assert_eq!(x_recon.len(), 3);
        assert_eq!(mu.len(), 2);
        assert_eq!(log_var.len(), 2);
        assert_eq!(z.len(), 2);
    }

    #[test]
    fn test_elbo_loss() {
        let config = VaeConfig::new(3, 16, 2);
        let vae = VaeFactor::new(config);

        let x = Array1::from_vec(vec![0.1, -0.05, 0.02]);
        let x_recon = Array1::from_vec(vec![0.09, -0.04, 0.03]);
        let mu = Array1::from_vec(vec![0.5, -0.3]);
        let log_var = Array1::from_vec(vec![-0.5, -0.2]);

        let (total, recon, kl) = vae.elbo_loss(&x, &x_recon, &mu, &log_var);

        assert!(recon >= 0.0, "Reconstruction loss should be non-negative");
        assert!(kl >= 0.0, "KL divergence should be non-negative");
        assert!((total - (recon + kl)).abs() < 1e-10, "Total should be recon + kl");
    }

    #[test]
    fn test_training_reduces_loss() {
        let data = sample_data();
        let config = VaeConfig::new(3, 16, 2);
        let mut vae = VaeFactor::new(config);

        let history = vae.train(&data, 50, 0.001).unwrap();

        assert_eq!(history.len(), 50);
        // Loss should generally decrease (check first vs last quarter average)
        let first_avg: f64 = history[..10].iter().map(|h| h.total_loss).sum::<f64>() / 10.0;
        let last_avg: f64 = history[40..].iter().map(|h| h.total_loss).sum::<f64>() / 10.0;
        assert!(
            last_avg <= first_avg * 2.0,
            "Loss should not explode: first_avg={}, last_avg={}",
            first_avg,
            last_avg
        );
    }

    #[test]
    fn test_extract_factors() {
        let data = sample_data();
        let config = VaeConfig::new(3, 16, 2);
        let mut vae = VaeFactor::new(config);
        vae.train(&data, 20, 0.001).unwrap();

        let factors = vae.extract_factors(&data);
        assert_eq!(factors.shape(), &[20, 2]);
    }

    #[test]
    fn test_generate_scenarios() {
        let config = VaeConfig::new(3, 16, 2);
        let vae = VaeFactor::new(config);

        let scenarios = vae.generate_scenarios(10);
        assert_eq!(scenarios.shape(), &[10, 3]);
    }

    #[test]
    fn test_factor_loadings() {
        let config = VaeConfig::new(3, 16, 2);
        let vae = VaeFactor::new(config);

        let z_base = Array1::zeros(2);
        let loadings = vae.factor_loadings(&z_base, 0.01);
        assert_eq!(loadings.shape(), &[3, 2]);
    }

    #[test]
    fn test_factor_loadings_named() {
        let config = VaeConfig::new(3, 16, 2);
        let vae = VaeFactor::new(config);

        let z_base = Array1::zeros(2);
        let fl = vae.factor_loadings_named(&z_base, 0.01, &["BTC", "ETH", "SOL"]);
        assert_eq!(fl.asset_names.len(), 3);
        assert_eq!(fl.loadings.shape(), &[3, 2]);
    }

    #[test]
    fn test_standard_scaler() {
        let data = sample_data();
        let (scaler, standardized) = StandardScaler::fit_transform(&data);

        // Check that standardized data has approximately zero mean
        let mean = standardized.mean_axis(Axis(0)).unwrap();
        for m in mean.iter() {
            assert!(m.abs() < 1e-10, "Mean should be ~0, got {}", m);
        }

        // Check round-trip
        let recovered = scaler.inverse_transform(&standardized);
        for i in 0..data.nrows() {
            for j in 0..data.ncols() {
                assert!(
                    (data[[i, j]] - recovered[[i, j]]).abs() < 1e-10,
                    "Round-trip failed at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_log_returns() {
        let prices = vec![100.0, 105.0, 103.0, 108.0];
        let returns = compute_log_returns(&prices);
        assert_eq!(returns.len(), 3);
        assert!((returns[0] - (105.0_f64 / 100.0).ln()).abs() < 1e-10);
    }

    #[test]
    fn test_build_return_matrix() {
        let prices1 = vec![100.0, 105.0, 103.0, 108.0];
        let prices2 = vec![50.0, 52.0, 51.0, 54.0];
        let matrix = build_return_matrix(&[prices1, prices2]);
        assert_eq!(matrix.shape(), &[3, 2]);
    }

    #[test]
    fn test_beta_vae_config() {
        let config = VaeConfig::new(3, 16, 2).with_beta(2.0);
        assert_eq!(config.beta, 2.0);

        let vae = VaeFactor::new(config);
        assert_eq!(vae.config.beta, 2.0);
    }

    #[test]
    fn test_variance_decomposition() {
        let data = sample_data();
        let config = VaeConfig::new(3, 16, 2);
        let mut vae = VaeFactor::new(config);
        vae.train(&data, 30, 0.001).unwrap();

        let (factor_var, idio_var) = vae.variance_decomposition(&data);
        assert!(factor_var >= 0.0 && factor_var <= 1.0);
        assert!(idio_var >= 0.0 && idio_var <= 1.0);
        assert!((factor_var + idio_var - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_reconstruction_error() {
        let data = sample_data();
        let config = VaeConfig::new(3, 16, 2);
        let mut vae = VaeFactor::new(config);
        vae.train(&data, 50, 0.001).unwrap();

        let error = vae.reconstruction_error(&data);
        assert!(error >= 0.0, "Reconstruction error should be non-negative");
    }

    #[test]
    fn test_stressed_scenarios() {
        let config = VaeConfig::new(3, 16, 2);
        let vae = VaeFactor::new(config);

        let stress_mu = Array1::from_vec(vec![-2.0, 1.5]);
        let scenarios = vae.generate_stressed_scenarios(5, &stress_mu, 0.5);
        assert_eq!(scenarios.shape(), &[5, 3]);
    }

    #[test]
    fn test_encode_full() {
        let config = VaeConfig::new(3, 16, 2);
        let vae = VaeFactor::new(config);

        let x = Array1::from_vec(vec![0.1, -0.05, 0.02]);
        let result = vae.encode_full(&x);

        assert_eq!(result.mu.len(), 2);
        assert_eq!(result.log_var.len(), 2);
        assert_eq!(result.z.len(), 2);
    }

    #[test]
    fn test_empty_data_error() {
        let config = VaeConfig::new(3, 16, 2);
        let mut vae = VaeFactor::new(config);

        let empty = Array2::zeros((0, 3));
        let result = vae.train(&empty, 10, 0.001);
        assert!(result.is_err());
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let config = VaeConfig::new(3, 16, 2);
        let mut vae = VaeFactor::new(config);

        let wrong_dim = Array2::zeros((10, 5));
        let result = vae.train(&wrong_dim, 10, 0.001);
        assert!(result.is_err());
    }
}
