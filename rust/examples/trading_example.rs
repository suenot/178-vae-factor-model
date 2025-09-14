//! # VAE Factor Model Trading Example
//!
//! This example demonstrates:
//! 1. Fetching multi-asset crypto data from Bybit (BTCUSDT, ETHUSDT, SOLUSDT)
//! 2. Training a VAE on multi-asset returns
//! 3. Extracting latent factors
//! 4. Generating synthetic market scenarios
//! 5. Analyzing factor loadings per asset

use ndarray::Array1;
use vae_factor_model::{
    build_return_matrix, BybitDataLoader, StandardScaler, VaeConfig,
    VaeFactor,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== VAE Factor Model for Crypto Markets ===\n");

    // ── Step 1: Fetch data from Bybit ─────────────────────────────────────
    println!("Step 1: Fetching data from Bybit...");
    let symbols = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT"];
    let loader = BybitDataLoader::new();

    let mut all_prices: Vec<Vec<f64>> = Vec::new();

    for symbol in &symbols {
        match loader.fetch_close_prices(symbol, "60", 200).await {
            Ok(prices) => {
                println!(
                    "  {} - fetched {} candles (latest close: {:.2})",
                    symbol,
                    prices.len(),
                    prices.last().unwrap_or(&0.0)
                );
                all_prices.push(prices);
            }
            Err(e) => {
                println!("  {} - failed to fetch: {}. Using synthetic data.", symbol, e);
                // Generate synthetic prices as fallback
                let mut rng = rand::thread_rng();
                use rand::Rng;
                let mut prices = vec![100.0_f64];
                for _ in 1..200 {
                    let ret: f64 = rng.gen_range(-0.03..0.03);
                    prices.push(prices.last().unwrap() * (1.0 + ret));
                }
                all_prices.push(prices);
            }
        }
    }

    // ── Step 2: Build return matrix and standardize ───────────────────────
    println!("\nStep 2: Building return matrix...");
    let return_matrix = build_return_matrix(&all_prices);
    println!(
        "  Return matrix shape: {} time steps x {} assets",
        return_matrix.nrows(),
        return_matrix.ncols()
    );

    let (scaler, standardized) = StandardScaler::fit_transform(&return_matrix);
    println!("  Data standardized (zero mean, unit variance)");

    // Print return statistics
    for (i, symbol) in symbols.iter().enumerate() {
        let col = return_matrix.column(i);
        let mean = col.mean().unwrap_or(0.0);
        let std = col.std(0.0);
        println!("  {} - mean: {:.6}, std: {:.6}", symbol, mean, std);
    }

    // ── Step 3: Configure and train VAE ───────────────────────────────────
    println!("\nStep 3: Training VAE factor model...");
    let input_dim = symbols.len();
    let hidden_dim = 16;
    let latent_dim = 2; // 2 latent factors

    let config = VaeConfig::new(input_dim, hidden_dim, latent_dim).with_beta(1.5);
    let mut vae = VaeFactor::new(config);

    let epochs = 100;
    let lr = 0.001;
    let history = vae.train(&standardized, epochs, lr)?;

    println!("  Training complete!");
    println!(
        "  Initial loss: {:.6} (recon: {:.6}, KL: {:.6})",
        history[0].total_loss, history[0].recon_loss, history[0].kl_loss
    );
    println!(
        "  Final loss:   {:.6} (recon: {:.6}, KL: {:.6})",
        history.last().unwrap().total_loss,
        history.last().unwrap().recon_loss,
        history.last().unwrap().kl_loss
    );

    // ── Step 4: Extract latent factors ────────────────────────────────────
    println!("\nStep 4: Extracting latent factors...");
    let factors = vae.extract_factors(&standardized);
    println!(
        "  Factor matrix shape: {} x {}",
        factors.nrows(),
        factors.ncols()
    );

    // Print factor statistics
    for j in 0..latent_dim {
        let col = factors.column(j);
        let mean = col.mean().unwrap_or(0.0);
        let std = col.std(0.0);
        let min = col.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!(
            "  Factor {} - mean: {:.4}, std: {:.4}, range: [{:.4}, {:.4}]",
            j, mean, std, min, max
        );
    }

    // Print latest factor values
    let n = factors.nrows();
    if n > 0 {
        println!("\n  Latest 5 factor observations:");
        let start = if n > 5 { n - 5 } else { 0 };
        for i in start..n {
            let vals: Vec<String> = (0..latent_dim)
                .map(|j| format!("{:.4}", factors[[i, j]]))
                .collect();
            println!("    t-{}: [{}]", n - i - 1, vals.join(", "));
        }
    }

    // ── Step 5: Generate synthetic scenarios ──────────────────────────────
    println!("\nStep 5: Generating synthetic market scenarios...");
    let n_scenarios = 10;
    let scenarios_std = vae.generate_scenarios(n_scenarios);
    let scenarios = scaler.inverse_transform(&scenarios_std);

    println!("  Generated {} scenarios (in return space):", n_scenarios);
    for i in 0..n_scenarios.min(5) {
        let vals: Vec<String> = symbols
            .iter()
            .enumerate()
            .map(|(j, sym)| format!("{}: {:.4}%", sym, scenarios[[i, j]] * 100.0))
            .collect();
        println!("    Scenario {}: {}", i + 1, vals.join(", "));
    }

    // Stressed scenarios (e.g., market crash)
    println!("\n  Stressed scenarios (crash regime):");
    let stress_mu = Array1::from_vec(vec![-2.0; latent_dim]);
    let stressed_std = vae.generate_stressed_scenarios(5, &stress_mu, 0.3);
    let stressed = scaler.inverse_transform(&stressed_std);

    for i in 0..5 {
        let vals: Vec<String> = symbols
            .iter()
            .enumerate()
            .map(|(j, sym)| format!("{}: {:.4}%", sym, stressed[[i, j]] * 100.0))
            .collect();
        println!("    Crash {}: {}", i + 1, vals.join(", "));
    }

    // ── Step 6: Factor loading analysis ───────────────────────────────────
    println!("\nStep 6: Analyzing factor loadings...");
    let z_base = Array1::zeros(latent_dim);
    let sym_refs: Vec<&str> = symbols.iter().map(|s| s.as_ref()).collect();
    let fl = vae.factor_loadings_named(&z_base, 0.01, &sym_refs);
    println!("{}", fl);

    // Interpret the loadings
    println!("  Interpretation:");
    for j in 0..latent_dim {
        let col = fl.loadings.column(j);
        let max_idx = col
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let all_same_sign = col.iter().all(|&v| v > 0.0) || col.iter().all(|&v| v < 0.0);

        if all_same_sign {
            println!(
                "    Factor {}: Market-wide factor (all assets move together), strongest on {}",
                j, symbols[max_idx]
            );
        } else {
            println!(
                "    Factor {}: Relative value factor (assets move in opposite directions), strongest on {}",
                j, symbols[max_idx]
            );
        }
    }

    // ── Step 7: Variance decomposition ────────────────────────────────────
    println!("\nStep 7: Risk decomposition...");
    let (factor_var, idio_var) = vae.variance_decomposition(&standardized);
    println!("  Factor-driven variance:     {:.1}%", factor_var * 100.0);
    println!("  Idiosyncratic variance:     {:.1}%", idio_var * 100.0);

    let recon_error = vae.reconstruction_error(&standardized);
    println!("  Mean reconstruction error:  {:.6}", recon_error);

    println!("\n=== Done ===");
    Ok(())
}
