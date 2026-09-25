"""
Institutional Reinforcement Learning Trading Terminal for QuantTerminal.
Powered by TensorTrade (1.0.4) & Deep Neural Network Agents:
- TensorTrade OMS Architecture: Simulated Exchange, Wallets, Portfolio, and DataFeed
- Technical Feature Stream Pipeline: Price, Return, SMA Ratio, RSI, MACD, and Volume
- Configurable Action Schemes: Discrete proportional allocation (Buy / Hold / Sell)
- Configurable Reward Schemes: Simple PnL, Risk-Adjusted Returns (Sortino/Sharpe), and Position-Based Returns (PBR)
- Deep RL Agent: Deep Q-Network (DQN) with Replay Buffer, Target Network, and Epsilon-Greedy Schedule
- 8-Tab Institutional Modular Suite:
    1. 🎮 Training Workbench & Real-Time Telemetry
    2. 📈 Out-of-Sample Validation & Equity Curve
    3. 🎯 Policy Execution Markers & Action Distribution
    4. 🧠 Q-Value Surface & State-Action Inspection
    5. 🔬 Reward Engineering Comparison
    6. 🏆 RL vs Quantitative Strategy Tournament
    7. 📚 Reinforcement Learning & TensorTrade Methodology
    8. 📋 Order Execution Log & CSV Research Tearsheet
"""

import math
import random
import datetime
import warnings
from collections import deque
from typing import Dict, List, Tuple, Any, Optional

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from decimal import Decimal

# TensorTrade Imports
import tensortrade.env.default as default
from tensortrade.feed.core import Stream, DataFeed
from tensortrade.oms.instruments import Instrument, TradingPair
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order

from utils.helper import (
    inject_custom_theme,
    load_data,
    drop_holiday_nans,
    fetch_stocks,
    CURRENCY_SYMBOLS,
    _fmt_num,
    _fmt_money,
    _fmt_pct
)
from utils.sidebar import render_sidebar

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Reinforcement Learning - QuantTerminal",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_custom_theme()

# ---------------------------------------------------------
# Data Caching Functions
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_processed_data(ticker_symbol: str, period_str: str, interval_str: str) -> pd.DataFrame:
    df_raw = load_data(ticker_symbol, period=period_str, interval=interval_str)
    return drop_holiday_nans(df_raw)

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
ticker, company, exchange_name, period, interval, region = render_sidebar()
currency_code = "INR" if region == "India" else "USD"
currency_sym = CURRENCY_SYMBOLS.get(currency_code, "$")

# ---------------------------------------------------------
# Header & Context Banner
# ---------------------------------------------------------
st.title("🤖 Reinforcement Learning Trading Terminal")
st.caption(
    "Algorithmic reinforcement learning platform powered by **TensorTrade**, Deep Q-Networks (DQN), "
    "and risk-adjusted reward engineering evaluated on institutional out-of-sample partitions."
)

st.info(
    "ℹ️ **TensorTrade Architecture:** The agent observes market state through a multi-stream `DataFeed` and executes discrete "
    "portfolio reallocation orders through a simulated `Exchange`. Policy learning strictly separates In-Sample training (70%) "
    "from Out-of-Sample validation (30%) to eliminate look-ahead bias."
)

st.markdown("---")

# ---------------------------------------------------------
# Feature Engineering Function
# ---------------------------------------------------------
def prepare_rl_features(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    c = df["Close"]
    df["Return"] = np.log(c / c.shift(1)).fillna(0.0)
    df["SMA_10"] = c.rolling(10).mean().bfill()
    df["SMA_30"] = c.rolling(30).mean().bfill()
    df["SMA_Ratio"] = (df["SMA_10"] / df["SMA_30"]) - 1.0
    
    # RSI 14
    delta = c.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df["RSI"] = 100.0 - (100.0 / (1.0 + rs))
    df["RSI"] = df["RSI"].bfill().fillna(50.0)
    df["RSI_Norm"] = (df["RSI"] - 50.0) / 50.0

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["MACD"] = (ema12 - ema26) / c
    
    # Volatility Ratio
    if "Volume" in df.columns and (df["Volume"] > 0).any():
        v_mean = df["Volume"].rolling(20).mean().bfill()
        df["Vol_Ratio"] = ((df["Volume"] / (v_mean + 1e-10)) - 1.0).clip(-2.0, 3.0)
    else:
        df["Vol_Ratio"] = 0.0

    # Normalized price relative to rolling 30-day mean
    df["Price_Norm"] = (c / df["SMA_30"]) - 1.0

    return df

# ---------------------------------------------------------
# Load Data & Split
# ---------------------------------------------------------
df_raw = get_processed_data(ticker, period, interval)

if df_raw.empty or "Close" not in df_raw.columns or len(df_raw) < 60:
    st.error(f"Insufficient historical bars available for **{ticker}** ({len(df_raw) if not df_raw.empty else 0} bars). Please select a different ticker or longer timeframe.")
    st.stop()

df_feat = prepare_rl_features(df_raw)
N_total = len(df_feat)
N_train = int(0.70 * N_total)

df_train = df_feat.iloc[:N_train].copy()
df_test = df_feat.iloc[N_train:].copy()

st.caption(
    f"📊 **Data Partition:** `{N_total}` observations | **In-Sample Calibration (70%):** `{len(df_train)}` bars "
    f"(`{df_train.index[0].strftime('%Y-%m-%d')}` to `{df_train.index[-1].strftime('%Y-%m-%d')}`) | "
    f"**Out-of-Sample Test (30%):** `{len(df_test)}` bars "
    f"(`{df_test.index[0].strftime('%Y-%m-%d')}` to `{df_test.index[-1].strftime('%Y-%m-%d')}`)."
)

st.markdown("---")

# ---------------------------------------------------------
# RL Agent & Environment Configuration Grid
# ---------------------------------------------------------
st.subheader("⚙️ TensorTrade Environment & Neural Agent Configuration")

c_cfg1, c_cfg2, c_cfg3, c_cfg4 = st.columns(4)
with c_cfg1:
    initial_cash = st.selectbox(
        "Initial Bankroll",
        [10000.0, 50000.0, 100000.0, 500000.0],
        index=2,
        format_func=lambda x: f"{currency_sym}{x:,.0f}",
        key="rl_init_cash_select"
    )

with c_cfg2:
    reward_scheme_choice = st.selectbox(
        "TensorTrade Reward Scheme",
        ["Simple Profit (PnL)", "Risk-Adjusted (Sortino / Vol Penalty)", "Position-Based Return (PBR)"],
        index=0,
        key="rl_reward_select"
    )

with c_cfg3:
    agent_arch = st.selectbox(
        "RL Agent Algorithm",
        [
            "Proximal Policy Optimization (PPO - Continuous)",
            "Soft Actor-Critic (SAC - Maximum Entropy Continuous)",
            "Deep Q-Network (DQN - Discrete)",
            "Deep Recurrent Q-Network (DRQN - Recurrent)",
            "Random Exploration Baseline"
        ],
        index=0,
        key="rl_agent_arch_select"
    )

with c_cfg4:
    training_preset = st.selectbox(
        "Training Preset",
        ["Fast Interactive (15 Episodes)", "Standard (30 Episodes)", "Deep Training (60 Episodes)"],
        index=0,
        key="rl_preset_select"
    )

n_episodes = 15 if "15" in training_preset else (30 if "30" in training_preset else 60)

c_hp1, c_hp2, c_hp3, c_hp4 = st.columns(4)
with c_hp1:
    lr = st.slider("Learning Rate (α)", 0.0001, 0.01, 0.001, step=0.0005, format="%.4f", key="rl_lr_slider")
with c_hp2:
    gamma = st.slider("Discount Factor (γ)", 0.80, 0.99, 0.95, step=0.01, key="rl_gamma_slider")
with c_hp3:
    comm_rate_bps = st.slider("Commission + Slippage (bps)", 0, 50, 10, step=5, key="rl_comm_slider") / 10000.0
with c_hp4:
    if "PPO" in agent_arch:
        ppo_clip_eps = st.slider("PPO Clipping (ε-clip)", 0.10, 0.30, 0.20, step=0.02, key="rl_ppo_clip_slider")
    elif "SAC" in agent_arch:
        sac_tau = st.slider("Polyak Target Rate (τ)", 0.01, 0.20, 0.05, step=0.01, key="rl_sac_tau_slider")
    elif "DRQN" in agent_arch:
        epsilon_decay = st.slider("Exploration Decay (ε-decay)", 0.90, 0.99, 0.95, step=0.01, key="rl_eps_decay_slider")
    else:
        epsilon_decay = st.slider("Exploration Decay (ε-decay)", 0.90, 0.99, 0.95, step=0.01, key="rl_eps_decay_slider")

st.markdown("---")

# ---------------------------------------------------------
# TensorTrade Environment Factory
# ---------------------------------------------------------
def create_tt_env(df_slice: pd.DataFrame, init_capital: float, r_scheme_str: str) -> Any:
    clean_curr = "".join(c for c in currency_code if c.isalnum())
    clean_sym = "".join(c for c in ticker if c.isalnum())

    base_inst = Instrument(clean_curr, 2, currency_code)
    stock_inst = Instrument(clean_sym, 2, ticker)
    pair_name = f"{clean_curr}/{clean_sym}"

    price_stream = Stream.source(list(df_slice["Close"].values), dtype="float").rename(pair_name)
    exchange = Exchange("simulated", service=execute_order)(price_stream)

    # Register fallback keys in _price_streams to prevent KeyError across dot/hyphen formatting differences
    pair_obj = TradingPair(base_inst, stock_inst)
    exchange._price_streams[str(pair_obj)] = price_stream
    exchange._price_streams[f"{clean_curr}/{clean_sym}"] = price_stream
    exchange._price_streams[f"{currency_code}/{ticker}"] = price_stream
    exchange._price_streams[f"{currency_code}-{ticker}"] = price_stream

    # Fail-safe quote_price patch in case TensorTrade queries unexpected variations
    orig_quote = exchange.quote_price
    def safe_quote(trading_pair):
        try:
            return orig_quote(trading_pair)
        except (KeyError, ValueError):
            return Decimal(str(price_stream.value))
    exchange.quote_price = safe_quote

    wallet_cash = Wallet(exchange, init_capital * base_inst)
    wallet_stock = Wallet(exchange, 0 * stock_inst)
    portfolio = Portfolio(base_inst, [wallet_cash, wallet_stock])

    # Construct feature streams
    p_stream = Stream.source(list(df_slice["Close"].values), dtype="float").rename("price")
    r_stream = Stream.source(list(df_slice["Return"].values), dtype="float").rename("return")
    sma_stream = Stream.source(list(df_slice["SMA_Ratio"].values), dtype="float").rename("sma_ratio")
    rsi_stream = Stream.source(list(df_slice["RSI_Norm"].values), dtype="float").rename("rsi_norm")
    macd_stream = Stream.source(list(df_slice["MACD"].values), dtype="float").rename("macd")
    vol_stream = Stream.source(list(df_slice["Vol_Ratio"].values), dtype="float").rename("vol_ratio")

    feed = DataFeed([p_stream, r_stream, sma_stream, rsi_stream, macd_stream, vol_stream])

    tt_reward = "risk-adjusted" if "Risk-Adjusted" in r_scheme_str else "simple"

    env = default.create(
        portfolio=portfolio,
        action_scheme="simple",
        reward_scheme=tt_reward,
        feed=feed
    )
    return env, portfolio

# ---------------------------------------------------------
# Continuous Proximal Policy Optimization (PPO) Actor-Critic Agent
# ---------------------------------------------------------
class ContinuousPPOAgent:
    """Actor-Critic Continuous Policy Optimization agent with GAE and clipped objective."""
    def __init__(self, state_dim: int, lr: float = 0.002, clip_eps: float = 0.2, gae_lambda: float = 0.95):
        self.state_dim = state_dim
        self.lr = lr
        self.clip_eps = clip_eps
        self.gae_lambda = gae_lambda
        
        np.random.seed(42)
        # Actor: state -> 32 -> 16 -> 1 (Sigmoid mean target exposure in [0, 1])
        self.w1_a = np.random.randn(state_dim, 32) * np.sqrt(2.0 / state_dim)
        self.b1_a = np.zeros((1, 32))
        self.w2_a = np.random.randn(32, 16) * np.sqrt(2.0 / 32)
        self.b2_a = np.zeros((1, 16))
        self.w3_a = np.random.randn(16, 1) * np.sqrt(2.0 / 16)
        self.b3_a = np.zeros((1, 1))
        self.log_std = -0.7 # std ~ 0.5 for controlled exploration

        # Critic: state -> 32 -> 16 -> 1 (Scalar State Value)
        self.w1_c = np.random.randn(state_dim, 32) * np.sqrt(2.0 / state_dim)
        self.b1_c = np.zeros((1, 32))
        self.w2_c = np.random.randn(32, 16) * np.sqrt(2.0 / 32)
        self.b2_c = np.zeros((1, 16))
        self.w3_c = np.random.randn(16, 1) * np.sqrt(2.0 / 16)
        self.b3_c = np.zeros((1, 1))

    def forward_actor(self, state: np.ndarray):
        s = np.atleast_2d(state)
        z1 = np.maximum(0, np.dot(s, self.w1_a) + self.b1_a)
        z2 = np.maximum(0, np.dot(z1, self.w2_a) + self.b2_a)
        raw_out = np.dot(z2, self.w3_a) + self.b3_a
        mu = 1.0 / (1.0 + np.exp(-np.clip(raw_out, -10.0, 10.0)))
        std = float(np.exp(self.log_std))
        return mu, std, z1, z2

    def forward_critic(self, state: np.ndarray):
        s = np.atleast_2d(state)
        z1 = np.maximum(0, np.dot(s, self.w1_c) + self.b1_c)
        z2 = np.maximum(0, np.dot(z1, self.w2_c) + self.b2_c)
        val = np.dot(z2, self.w3_c) + self.b3_c
        return val, z1, z2

    def forward(self, state: np.ndarray) -> np.ndarray:
        # Pseudo-Q compatibility for visualization inspection
        mu, _, _, _ = self.forward_actor(state)
        v, _, _ = self.forward_critic(state)
        alloc = float(mu[0, 0])
        val = float(v[0, 0])
        return np.array([val * (1.0 - alloc), val * alloc, val * (1.0 - alloc)])

    def get_action(self, state: np.ndarray, deterministic: bool = False):
        mu, std, _, _ = self.forward_actor(state)
        mu_val = float(mu[0, 0])
        val = float(self.forward_critic(state)[0][0, 0])
        if deterministic:
            return float(np.clip(mu_val, 0.0, 1.0)), 0.0, val
        
        noise = np.random.normal(0, std)
        act = float(np.clip(mu_val + noise, 0.01, 0.99))
        var = std ** 2
        log_prob = -0.5 * (((act - mu_val) ** 2) / (var + 1e-8) + np.log(2 * np.pi * var + 1e-8))
        return act, float(log_prob), val

    def train_trajectory(self, states, actions, rewards, next_states, dones, old_log_probs, values, gamma: float = 0.95):
        T = len(states)
        if T < 2:
            return 0.0, 0.0, 0.0
        
        advantages = np.zeros(T)
        returns = np.zeros(T)
        gae = 0.0
        
        last_val, _, _ = self.forward_critic(next_states[-1])
        last_val = float(last_val[0, 0])
        
        for t in reversed(range(T)):
            next_non_terminal = 1.0 - float(dones[t])
            next_v = last_val if t == T - 1 else values[t + 1]
            delta = rewards[t] + gamma * next_v * next_non_terminal - values[t]
            gae = delta + gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
            
        adv_norm = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        states_arr = np.array(states)
        actions_arr = np.array(actions).reshape(-1, 1)
        returns_arr = np.array(returns).reshape(-1, 1)
        adv_arr = adv_norm.reshape(-1, 1)
        old_lp_arr = np.array(old_log_probs).reshape(-1, 1)

        pol_loss = 0.0
        val_loss = 0.0
        std_pred = float(np.exp(self.log_std))

        for _ in range(3):
            # Critic gradient
            v_pred, z1_c, z2_c = self.forward_critic(states_arr)
            grad_crit = 2.0 * (v_pred - returns_arr) / T
            val_loss = float(np.mean((v_pred - returns_arr) ** 2))
            
            dw3_c = np.dot(z2_c.T, grad_crit)
            db3_c = np.sum(grad_crit, axis=0, keepdims=True)
            dz2_c = np.dot(grad_crit, self.w3_c.T) * (z2_c > 0)
            dw2_c = np.dot(z1_c.T, dz2_c)
            db2_c = np.sum(dz2_c, axis=0, keepdims=True)
            dz1_c = np.dot(dz2_c, self.w2_c.T) * (z1_c > 0)
            dw1_c = np.dot(states_arr.T, dz1_c)
            db1_c = np.sum(dz1_c, axis=0, keepdims=True)

            self.w3_c -= self.lr * np.clip(dw3_c, -2.0, 2.0)
            self.b3_c -= self.lr * np.clip(db3_c, -2.0, 2.0)
            self.w2_c -= self.lr * np.clip(dw2_c, -2.0, 2.0)
            self.b2_c -= self.lr * np.clip(db2_c, -2.0, 2.0)
            self.w1_c -= self.lr * np.clip(dw1_c, -2.0, 2.0)
            self.b1_c -= self.lr * np.clip(db1_c, -2.0, 2.0)

            # Actor gradient
            mu_pred, std_pred, z1_a, z2_a = self.forward_actor(states_arr)
            var_pred = std_pred ** 2
            curr_log_prob = -0.5 * (((actions_arr - mu_pred) ** 2) / (var_pred + 1e-8) + np.log(2 * np.pi * var_pred + 1e-8))
            ratio = np.exp(np.clip(curr_log_prob - old_lp_arr, -10.0, 10.0))
            surr1 = ratio * adv_arr
            surr2 = np.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_arr
            clip_mask = (surr1 <= surr2) | ((ratio > 1.0 - self.clip_eps) & (ratio < 1.0 + self.clip_eps))
            
            d_lp_d_mu = (actions_arr - mu_pred) / (var_pred + 1e-8)
            d_loss_d_mu = -ratio * adv_arr * d_lp_d_mu * clip_mask / T
            d_raw = d_loss_d_mu * (mu_pred * (1.0 - mu_pred))
            
            dw3_a = np.dot(z2_a.T, d_raw)
            db3_a = np.sum(d_raw, axis=0, keepdims=True)
            dz2_a = np.dot(d_raw, self.w3_a.T) * (z2_a > 0)
            dw2_a = np.dot(z1_a.T, dz2_a)
            db2_a = np.sum(dz2_a, axis=0, keepdims=True)
            dz1_a = np.dot(dz2_a, self.w2_a.T) * (z1_a > 0)
            dw1_a = np.dot(states_arr.T, dz1_a)
            db1_a = np.sum(dz1_a, axis=0, keepdims=True)

            self.w3_a -= self.lr * np.clip(dw3_a, -2.0, 2.0)
            self.b3_a -= self.lr * np.clip(db3_a, -2.0, 2.0)
            self.w2_a -= self.lr * np.clip(dw2_a, -2.0, 2.0)
            self.b2_a -= self.lr * np.clip(db2_a, -2.0, 2.0)
            self.w1_a -= self.lr * np.clip(dw1_a, -2.0, 2.0)
            self.b1_a -= self.lr * np.clip(db1_a, -2.0, 2.0)

            pol_loss = float(-np.mean(np.minimum(surr1, surr2)))

        entropy = float(0.5 + 0.5 * np.log(2 * np.pi * (std_pred**2) + 1e-8))
        return pol_loss, val_loss, entropy

# ---------------------------------------------------------
# Soft Actor-Critic (SAC) Continuous Maximum Entropy Agent
# ---------------------------------------------------------
class SoftActorCriticAgent:
    """
    Institutional Soft Actor-Critic (SAC) Agent for continuous portfolio allocation:
    - Twin Q-Critics (Q1, Q2) to prevent value overestimation
    - Soft Target Critics with Polyak smoothing updates (τ)
    - Reparameterized Gaussian-Sigmoid Actor w_t in [0.0, 1.0] with analytical Jacobian correction
    - Dual entropy gradient for automatic temperature (α) adaptation
    """
    def __init__(self, state_dim: int, lr: float = 0.001, gamma: float = 0.95, tau: float = 0.05, target_entropy: float = -1.0):
        self.state_dim = state_dim
        self.action_dim = 1
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = target_entropy
        
        self.log_alpha = 0.0 # Initial alpha = exp(0) = 1.0
        self.lr_alpha = lr
        
        np.random.seed(42)
        # Actor network: state (6) -> 32 -> 16 -> (mu, log_std)
        self.w1_a = np.random.randn(state_dim, 32) * np.sqrt(2.0 / state_dim)
        self.b1_a = np.zeros((1, 32))
        self.w2_a = np.random.randn(32, 16) * np.sqrt(2.0 / 32)
        self.b2_a = np.zeros((1, 16))
        self.w_mu = np.random.randn(16, 1) * np.sqrt(2.0 / 16)
        self.b_mu = np.zeros((1, 1))
        self.w_std = np.random.randn(16, 1) * np.sqrt(2.0 / 16)
        self.b_std = np.zeros((1, 1))

        # Critic 1: concat(s, a) (7) -> 32 -> 16 -> 1
        in_dim = state_dim + 1
        self.w1_q1 = np.random.randn(in_dim, 32) * np.sqrt(2.0 / in_dim)
        self.b1_q1 = np.zeros((1, 32))
        self.w2_q1 = np.random.randn(32, 16) * np.sqrt(2.0 / 32)
        self.b2_q1 = np.zeros((1, 16))
        self.w3_q1 = np.random.randn(16, 1) * np.sqrt(2.0 / 16)
        self.b3_q1 = np.zeros((1, 1))

        # Critic 2: concat(s, a) (7) -> 32 -> 16 -> 1
        self.w1_q2 = np.random.randn(in_dim, 32) * np.sqrt(2.0 / in_dim)
        self.b1_q2 = np.zeros((1, 32))
        self.w2_q2 = np.random.randn(32, 16) * np.sqrt(2.0 / 32)
        self.b2_q2 = np.zeros((1, 16))
        self.w3_q2 = np.random.randn(16, 1) * np.sqrt(2.0 / 16)
        self.b3_q2 = np.zeros((1, 1))

        # Target Critics (Polyak targets)
        self.w1_t1, self.b1_t1 = self.w1_q1.copy(), self.b1_q1.copy()
        self.w2_t1, self.b2_t1 = self.w2_q1.copy(), self.b2_q1.copy()
        self.w3_t1, self.b3_t1 = self.w3_q1.copy(), self.b3_q1.copy()

        self.w1_t2, self.b1_t2 = self.w1_q2.copy(), self.b1_q2.copy()
        self.w2_t2, self.b2_t2 = self.w2_q2.copy(), self.b2_q2.copy()
        self.w3_t2, self.b3_t2 = self.w3_q2.copy(), self.b3_q2.copy()

    @property
    def alpha(self) -> float:
        return float(np.exp(np.clip(self.log_alpha, -4.0, 2.0)))

    def forward_actor(self, state: np.ndarray):
        s = np.atleast_2d(state)
        z1 = np.maximum(0, np.dot(s, self.w1_a) + self.b1_a)
        z2 = np.maximum(0, np.dot(z1, self.w2_a) + self.b2_a)
        mu = np.dot(z2, self.w_mu) + self.b_mu
        log_std = np.clip(np.dot(z2, self.w_std) + self.b_std, -2.5, 0.5)
        std = np.exp(log_std)
        return mu, std, log_std, z1, z2

    def sample_action(self, state: np.ndarray, deterministic: bool = False):
        mu, std, log_std, _, _ = self.forward_actor(state)
        if deterministic:
            u = mu
            a = 1.0 / (1.0 + np.exp(-np.clip(u, -10.0, 10.0)))
            log_prob = np.zeros_like(a)
            return a, log_prob, u
        
        eps = np.random.randn(*mu.shape)
        u = mu + std * eps
        a = 1.0 / (1.0 + np.exp(-np.clip(u, -10.0, 10.0)))
        gauss_lp = -0.5 * (((u - mu) / (std + 1e-8)) ** 2 + 2.0 * log_std + np.log(2.0 * np.pi))
        log_prob = gauss_lp - np.log(a * (1.0 - a) + 1e-7)
        return a, log_prob, u

    def get_action(self, state: np.ndarray, deterministic: bool = False):
        a, lp, _ = self.sample_action(state, deterministic=deterministic)
        act = float(np.clip(a[0, 0], 0.0, 1.0))
        q1_val = float(self.forward_q(self.w1_q1, self.b1_q1, self.w2_q1, self.b2_q1, self.w3_q1, self.b3_q1, state, np.array([[act]]))[0][0, 0])
        return act, float(lp[0, 0]), q1_val

    def forward_q(self, w1, b1, w2, b2, w3, b3, state: np.ndarray, action: np.ndarray):
        s = np.atleast_2d(state)
        a = np.atleast_2d(action)
        x = np.hstack([s, a])
        z1 = np.maximum(0, np.dot(x, w1) + b1)
        z2 = np.maximum(0, np.dot(z1, w2) + b2)
        q = np.dot(z2, w3) + b3
        return q, z1, z2, x

    def forward(self, state: np.ndarray) -> np.ndarray:
        s = np.atleast_2d(state)
        act_buy = np.ones((len(s), 1))
        act_hold = np.ones((len(s), 1)) * 0.5
        act_sell = np.zeros((len(s), 1))
        q_buy, _, _, _ = self.forward_q(self.w1_q1, self.b1_q1, self.w2_q1, self.b2_q1, self.w3_q1, self.b3_q1, s, act_buy)
        q_hold, _, _, _ = self.forward_q(self.w1_q1, self.b1_q1, self.w2_q1, self.b2_q1, self.w3_q1, self.b3_q1, s, act_hold)
        q_sell, _, _, _ = self.forward_q(self.w1_q1, self.b1_q1, self.w2_q1, self.b2_q1, self.w3_q1, self.b3_q1, s, act_sell)
        return np.array([float(q_hold[0, 0]), float(q_buy[0, 0]), float(q_sell[0, 0])])

    def train_step(self, b_states, b_actions, b_rewards, b_next_states, b_dones):
        N = len(b_states)
        s = np.array(b_states)
        a = np.array(b_actions).reshape(-1, 1)
        r = np.array(b_rewards).reshape(-1, 1)
        s_next = np.array(b_next_states)
        d = np.array(b_dones, dtype=float).reshape(-1, 1)

        # 1. Target Value computation using Target Twin Critics
        a_next, lp_next, _ = self.sample_action(s_next, deterministic=False)
        q1_targ, _, _, _ = self.forward_q(self.w1_t1, self.b1_t1, self.w2_t1, self.b2_t1, self.w3_t1, self.b3_t1, s_next, a_next)
        q2_targ, _, _, _ = self.forward_q(self.w1_t2, self.b1_t2, self.w2_t2, self.b2_t2, self.w3_t2, self.b3_t2, s_next, a_next)
        min_q_targ = np.minimum(q1_targ, q2_targ)
        y_target = r + self.gamma * (1.0 - d) * (min_q_targ - self.alpha * lp_next)

        # 2. Update Critic 1
        q1_pred, z1_q1, z2_q1, x_q1 = self.forward_q(self.w1_q1, self.b1_q1, self.w2_q1, self.b2_q1, self.w3_q1, self.b3_q1, s, a)
        grad_q1 = 2.0 * (q1_pred - y_target) / N
        dw3_q1 = np.dot(z2_q1.T, grad_q1)
        db3_q1 = np.sum(grad_q1, axis=0, keepdims=True)
        dz2_q1 = np.dot(grad_q1, self.w3_q1.T) * (z2_q1 > 0)
        dw2_q1 = np.dot(z1_q1.T, dz2_q1)
        db2_q1 = np.sum(dz2_q1, axis=0, keepdims=True)
        dz1_q1 = np.dot(dz2_q1, self.w2_q1.T) * (z1_q1 > 0)
        dw1_q1 = np.dot(x_q1.T, dz1_q1)
        db1_q1 = np.sum(dz1_q1, axis=0, keepdims=True)

        self.w3_q1 -= self.lr * np.clip(dw3_q1, -3.0, 3.0)
        self.b3_q1 -= self.lr * np.clip(db3_q1, -3.0, 3.0)
        self.w2_q1 -= self.lr * np.clip(dw2_q1, -3.0, 3.0)
        self.b2_q1 -= self.lr * np.clip(db2_q1, -3.0, 3.0)
        self.w1_q1 -= self.lr * np.clip(dw1_q1, -3.0, 3.0)
        self.b1_q1 -= self.lr * np.clip(db1_q1, -3.0, 3.0)

        # 3. Update Critic 2
        q2_pred, z1_q2, z2_q2, x_q2 = self.forward_q(self.w1_q2, self.b1_q2, self.w2_q2, self.b2_q2, self.w3_q2, self.b3_q2, s, a)
        grad_q2 = 2.0 * (q2_pred - y_target) / N
        dw3_q2 = np.dot(z2_q2.T, grad_q2)
        db3_q2 = np.sum(grad_q2, axis=0, keepdims=True)
        dz2_q2 = np.dot(grad_q2, self.w3_q2.T) * (z2_q2 > 0)
        dw2_q2 = np.dot(z1_q2.T, dz2_q2)
        db2_q2 = np.sum(dz2_q2, axis=0, keepdims=True)
        dz1_q2 = np.dot(dz2_q2, self.w2_q2.T) * (z1_q2 > 0)
        dw1_q2 = np.dot(x_q2.T, dz1_q2)
        db1_q2 = np.sum(dz1_q2, axis=0, keepdims=True)

        self.w3_q2 -= self.lr * np.clip(dw3_q2, -3.0, 3.0)
        self.b3_q2 -= self.lr * np.clip(db3_q2, -3.0, 3.0)
        self.w2_q2 -= self.lr * np.clip(dw2_q2, -3.0, 3.0)
        self.b2_q2 -= self.lr * np.clip(db2_q2, -3.0, 3.0)
        self.w1_q2 -= self.lr * np.clip(dw1_q2, -3.0, 3.0)
        self.b1_q2 -= self.lr * np.clip(db1_q2, -3.0, 3.0)

        # 4. Update Actor
        a_curr, lp_curr, u_curr = self.sample_action(s, deterministic=False)
        mu_curr, std_curr, log_std_curr, z1_a, z2_a = self.forward_actor(s)
        
        # dQ/da through Q1:
        q1_eval, z1_e, z2_e, x_e = self.forward_q(self.w1_q1, self.b1_q1, self.w2_q1, self.b2_q1, self.w3_q1, self.b3_q1, s, a_curr)
        dz2_dq = np.dot(np.ones_like(q1_eval), self.w3_q1.T) * (z2_e > 0)
        dz1_dq = np.dot(dz2_dq, self.w2_q1.T) * (z1_e > 0)
        dx_dq = np.dot(dz1_dq, self.w1_q1.T)
        dq_da = dx_dq[:, -1:]

        da_du = a_curr * (1.0 - a_curr)
        dJ_dmu = (-self.alpha * (1.0 - 2.0 * a_curr) - dq_da * da_du) / N
        eps_sample = (u_curr - mu_curr) / (std_curr + 1e-8)
        dJ_dlogstd = (-self.alpha - (self.alpha * (1.0 - 2.0 * a_curr) + dq_da * da_du) * eps_sample) / N

        dw_mu = np.dot(z2_a.T, dJ_dmu)
        db_mu = np.sum(dJ_dmu, axis=0, keepdims=True)
        dw_std = np.dot(z2_a.T, dJ_dlogstd)
        db_std = np.sum(dJ_dlogstd, axis=0, keepdims=True)

        dz2_a = (np.dot(dJ_dmu, self.w_mu.T) + np.dot(dJ_dlogstd, self.w_std.T)) * (z2_a > 0)
        dw2_a = np.dot(z1_a.T, dz2_a)
        db2_a = np.sum(dz2_a, axis=0, keepdims=True)
        dz1_a = np.dot(dz2_a, self.w2_a.T) * (z1_a > 0)
        dw1_a = np.dot(s.T, dz1_a)
        db1_a = np.sum(dz1_a, axis=0, keepdims=True)

        self.w_mu -= self.lr * np.clip(dw_mu, -2.0, 2.0)
        self.b_mu -= self.lr * np.clip(db_mu, -2.0, 2.0)
        self.w_std -= self.lr * np.clip(dw_std, -2.0, 2.0)
        self.b_std -= self.lr * np.clip(db_std, -2.0, 2.0)
        self.w2_a -= self.lr * np.clip(dw2_a, -2.0, 2.0)
        self.b2_a -= self.lr * np.clip(db2_a, -2.0, 2.0)
        self.w1_a -= self.lr * np.clip(dw1_a, -2.0, 2.0)
        self.b1_a -= self.lr * np.clip(db1_a, -2.0, 2.0)

        # 5. Automatic Temperature Tuning
        grad_log_alpha = -np.mean(lp_curr + self.target_entropy)
        self.log_alpha -= self.lr_alpha * np.clip(grad_log_alpha, -1.0, 1.0)
        self.log_alpha = float(np.clip(self.log_alpha, -4.0, 2.0))

        # 6. Polyak Soft Target Updates
        self.w1_t1 = self.tau * self.w1_q1 + (1.0 - self.tau) * self.w1_t1
        self.b1_t1 = self.tau * self.b1_q1 + (1.0 - self.tau) * self.b1_t1
        self.w2_t1 = self.tau * self.w2_q1 + (1.0 - self.tau) * self.w2_t1
        self.b2_t1 = self.tau * self.b2_q1 + (1.0 - self.tau) * self.b2_t1
        self.w3_t1 = self.tau * self.w3_q1 + (1.0 - self.tau) * self.w3_t1
        self.b3_t1 = self.tau * self.b3_q1 + (1.0 - self.tau) * self.b3_t1

        self.w1_t2 = self.tau * self.w1_q2 + (1.0 - self.tau) * self.w1_t2
        self.b1_t2 = self.tau * self.b1_q2 + (1.0 - self.tau) * self.b1_t2
        self.w2_t2 = self.tau * self.w2_q2 + (1.0 - self.tau) * self.w2_t2
        self.b2_t2 = self.tau * self.b2_q2 + (1.0 - self.tau) * self.b2_t2
        self.w3_t2 = self.tau * self.w3_q2 + (1.0 - self.tau) * self.w3_t2
        self.b3_t2 = self.tau * self.b3_q2 + (1.0 - self.tau) * self.b3_t2

        q_loss = float(0.5 * (np.mean((q1_pred - y_target)**2) + np.mean((q2_pred - y_target)**2)))
        pi_loss = float(np.mean(self.alpha * lp_curr - q1_eval))
        return q_loss, pi_loss, self.alpha
# ---------------------------------------------------------
class DRQNAgent:
    """Deep Recurrent Q-Network with internal recurrent hidden state for non-Markovian market regimes."""
    def __init__(self, state_dim: int, action_dim: int, lr: float = 0.001, hidden_dim: int = 16):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        
        np.random.seed(42)
        self.w_ih = np.random.randn(state_dim, hidden_dim) * np.sqrt(2.0 / state_dim)
        self.w_hh = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b_h = np.zeros((1, hidden_dim))
        
        self.w_out = np.random.randn(hidden_dim, action_dim) * np.sqrt(2.0 / hidden_dim)
        self.b_out = np.zeros((1, action_dim))
        self.h = np.zeros((1, hidden_dim))

    def reset_hidden(self):
        self.h = np.zeros((1, self.hidden_dim))

    def forward_step(self, state: np.ndarray, h_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        s = np.atleast_2d(state)
        z = np.dot(s, self.w_ih) + np.dot(h_prev, self.w_hh) + self.b_h
        h_new = np.tanh(z)
        q = np.dot(h_new, self.w_out) + self.b_out
        return q, h_new

    def forward(self, state: np.ndarray) -> np.ndarray:
        q, self.h = self.forward_step(state, self.h)
        return q

    def train_step(self, states: np.ndarray, targets: np.ndarray):
        s = np.atleast_2d(states)
        z = np.dot(s, self.w_ih) + self.b_h
        h = np.tanh(z)
        q_pred = np.dot(h, self.w_out) + self.b_out
        grad_out = 2.0 * (q_pred - targets) / len(s)
        
        dw_out = np.dot(h.T, grad_out)
        db_out = np.sum(grad_out, axis=0, keepdims=True)
        dh = np.dot(grad_out, self.w_out.T) * (1.0 - h ** 2)
        dw_ih = np.dot(s.T, dh)
        db_h = np.sum(dh, axis=0, keepdims=True)
        
        self.w_out -= self.lr * np.clip(dw_out, -3.0, 3.0)
        self.b_out -= self.lr * np.clip(db_out, -3.0, 3.0)
        self.w_ih -= self.lr * np.clip(dw_ih, -3.0, 3.0)
        self.b_h -= self.lr * np.clip(db_h, -3.0, 3.0)
        return float(np.mean((q_pred - targets) ** 2))

# ---------------------------------------------------------
# Lightweight Neural Deep Q-Network Agent
# ---------------------------------------------------------
class DeepQNetwork:
    """Multi-Layer Neural Q-Network with experience replay and target network."""
    def __init__(self, state_dim: int, action_dim: int, lr: float = 0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        
        # Initialize weights (He normal initialization)
        np.random.seed(42)
        h1 = 32
        h2 = 16
        self.w1 = np.random.randn(state_dim, h1) * np.sqrt(2.0 / state_dim)
        self.b1 = np.zeros((1, h1))
        self.w2 = np.random.randn(h1, h2) * np.sqrt(2.0 / h1)
        self.b2 = np.zeros((1, h2))
        self.w3 = np.random.randn(h2, action_dim) * np.sqrt(2.0 / h2)
        self.b3 = np.zeros((1, action_dim))

    def forward(self, state: np.ndarray) -> np.ndarray:
        s = np.atleast_2d(state)
        z1 = np.dot(s, self.w1) + self.b1
        a1 = np.maximum(0, z1) # ReLU
        z2 = np.dot(a1, self.w2) + self.b2
        a2 = np.maximum(0, z2) # ReLU
        q_vals = np.dot(a2, self.w3) + self.b3
        return q_vals

    def train_step(self, states: np.ndarray, targets: np.ndarray):
        s = np.atleast_2d(states)
        z1 = np.dot(s, self.w1) + self.b1
        a1 = np.maximum(0, z1)
        z2 = np.dot(a1, self.w2) + self.b2
        a2 = np.maximum(0, z2)
        q_pred = np.dot(a2, self.w3) + self.b3

        # Gradient of MSE loss: 2 * (q_pred - targets)
        grad_out = 2.0 * (q_pred - targets) / len(s)
        
        # Layer 3 grads
        dw3 = np.dot(a2.T, grad_out)
        db3 = np.sum(grad_out, axis=0, keepdims=True)

        # Layer 2 grads
        da2 = np.dot(grad_out, self.w3.T)
        dz2 = da2 * (z2 > 0)
        dw2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        # Layer 1 grads
        da1 = np.dot(dz2, self.w2.T)
        dz1 = da1 * (z1 > 0)
        dw1 = np.dot(s.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Update weights (SGD with clip)
        clip_val = 5.0
        self.w3 -= self.lr * np.clip(dw3, -clip_val, clip_val)
        self.b3 -= self.lr * np.clip(db3, -clip_val, clip_val)
        self.w2 -= self.lr * np.clip(dw2, -clip_val, clip_val)
        self.b2 -= self.lr * np.clip(db2, -clip_val, clip_val)
        self.w1 -= self.lr * np.clip(dw1, -clip_val, clip_val)
        self.b1 -= self.lr * np.clip(db1, -clip_val, clip_val)

        return float(np.mean((q_pred - targets) ** 2))

# ---------------------------------------------------------
# B15 Risk Guardrails & Kill-Switch Controls
# ---------------------------------------------------------
c_grd1, c_grd2, c_grd3 = st.columns(3)
with c_grd1:
    emergency_kill_switch = st.toggle("🚨 Emergency Kill-Switch (B15 Guardrail)", value=False, key="rl_emergency_kill_switch", help="Instantly forces all positions into 100% Cash.")
with c_grd2:
    circuit_breaker_mdd = st.slider("Max Drawdown Circuit Breaker (%)", 2.0, 25.0, 10.0, step=1.0, key="rl_circuit_breaker_slider") / 100.0
with c_grd3:
    auto_rebalance_days = st.selectbox("Action Rebalance Horizon", ["Every Session (Daily)", "Weekly (5 Days)", "Bi-Weekly (10 Days)"], index=0, key="rl_action_rebal_select")

st.markdown("---")

# ---------------------------------------------------------
# Training Execution Controller & Auto-Calibration
# ---------------------------------------------------------
col_btn1, col_btn2 = st.columns([3, 1])
with col_btn1:
    train_btn = st.button("▶ Train / Re-Train Reinforcement Learning Agent", type="primary", width="stretch", key="rl_train_trigger_btn")
with col_btn2:
    reset_btn = st.button("🔄 Reset Agent", width="stretch", key="rl_reset_btn")

if reset_btn:
    st.session_state["rl_trained"] = False
    st.session_state["rl_history"] = None
    st.session_state["rl_oos_res"] = None
    st.session_state["rl_q_net"] = None
    st.rerun()

# Auto-train on first load if not trained yet
trigger_training = train_btn or (not st.session_state.get("rl_trained", False))

# Session state initialization for persistence
if "rl_trained" not in st.session_state:
    st.session_state["rl_trained"] = False
    st.session_state["rl_history"] = None
    st.session_state["rl_oos_res"] = None
    st.session_state["rl_q_net"] = None

if trigger_training:
    with st.spinner(f"Calibrating TensorTrade environment & training {agent_arch} across {n_episodes} episodes..."):
        prog_bar = st.progress(0, text="Initializing TensorTrade OMS...")
        
        # State dimension: 6 features
        state_dim = 6
        action_dim = 3 # 0: Hold/Cash, 1: Buy (Long), 2: Sell (Flat)

        is_ppo = "PPO" in agent_arch
        is_sac = "SAC" in agent_arch
        is_continuous = is_ppo or is_sac
        is_drqn = "DRQN" in agent_arch
        is_baseline = "Baseline" in agent_arch

        if is_ppo:
            active_agent = ContinuousPPOAgent(
                state_dim=state_dim,
                lr=lr,
                clip_eps=ppo_clip_eps if "ppo_clip_eps" in locals() else 0.20,
                gae_lambda=0.95
            )
        elif is_sac:
            active_agent = SoftActorCriticAgent(
                state_dim=state_dim,
                lr=lr,
                gamma=gamma,
                tau=sac_tau if "sac_tau" in locals() else 0.05
            )
        elif is_drqn:
            active_agent = DRQNAgent(state_dim=state_dim, action_dim=action_dim, lr=lr)
        else:
            active_agent = DeepQNetwork(state_dim=state_dim, action_dim=action_dim, lr=lr)

        replay_buffer = deque(maxlen=2000)
        batch_size = 32

        eps = 1.0
        eps_min = 0.05

        history_episodes = []
        history_networth = []
        history_reward = []
        history_loss = []
        history_pol_loss = []
        history_val_loss = []
        history_entropy = []
        history_eps = []

        for ep in range(n_episodes):
            env, portfolio = create_tt_env(df_train, initial_cash, reward_scheme_choice)
            obs_tuple = env.reset()
            state = obs_tuple[0].flatten()[:state_dim]

            if is_drqn:
                active_agent.reset_hidden()

            ep_reward = 0.0
            ep_losses = []
            ep_pol_losses = []
            ep_alphas = []
            done = False
            step_count = 0

            # PPO trajectory storage
            ep_states, ep_acts, ep_rews, ep_next_s, ep_dones, ep_lps, ep_vals = [], [], [], [], [], [], []

            while not done and step_count < len(df_train) - 2:
                if is_continuous:
                    act_val, lp, v = active_agent.get_action(state, deterministic=False)
                    # Convert continuous target allocation to discrete order
                    curr_eq_val = float(portfolio.net_worth) - float(portfolio.wallets[0].balance.as_float())
                    curr_w = max(0.0, min(1.0, curr_eq_val / (float(portfolio.net_worth) + 1e-6)))
                    if act_val - curr_w > 0.05:
                        tt_action = 1 # Buy proportional
                    elif curr_w - act_val > 0.05:
                        tt_action = 2 # Sell proportional
                    else:
                        tt_action = 0 # Hold
                    action = tt_action
                elif is_baseline or np.random.rand() < eps:
                    action = np.random.randint(action_dim)
                    tt_action = action
                else:
                    q_vals = active_agent.forward(state)
                    action = int(np.argmax(q_vals))
                    tt_action = action

                step_res = env.step(tt_action)

                next_state = step_res[0].flatten()[:state_dim]
                reward = float(step_res[1])
                terminated = bool(step_res[2])
                truncated = bool(step_res[3])
                done = terminated or truncated

                # Friction penalty
                reward -= comm_rate_bps * (1.0 if action > 0 else 0.0)

                ep_reward += reward

                if is_ppo:
                    ep_states.append(state)
                    ep_acts.append(act_val)
                    ep_rews.append(reward)
                    ep_next_s.append(next_state)
                    ep_dones.append(done)
                    ep_lps.append(lp)
                    ep_vals.append(v)
                elif is_sac:
                    replay_buffer.append((state, act_val, reward, next_state, done))
                else:
                    replay_buffer.append((state, action, reward, next_state, done))

                state = next_state
                step_count += 1

                # Train step if enough memory for replay-based algorithms
                if is_sac and len(replay_buffer) >= batch_size:
                    minibatch = random.sample(replay_buffer, batch_size)
                    b_states = np.array([m[0] for m in minibatch])
                    b_actions = np.array([m[1] for m in minibatch])
                    b_rewards = np.array([m[2] for m in minibatch])
                    b_next_states = np.array([m[3] for m in minibatch])
                    b_dones = np.array([m[4] for m in minibatch])
                    q_l, p_l, alpha_val = active_agent.train_step(b_states, b_actions, b_rewards, b_next_states, b_dones)
                    ep_losses.append(q_l)
                    ep_pol_losses.append(p_l)
                    ep_alphas.append(alpha_val)
                elif not is_continuous and len(replay_buffer) >= batch_size and not is_baseline:
                    minibatch = random.sample(replay_buffer, batch_size)
                    b_states = np.array([m[0] for m in minibatch])
                    b_actions = np.array([m[1] for m in minibatch])
                    b_rewards = np.array([m[2] for m in minibatch])
                    b_next_states = np.array([m[3] for m in minibatch])
                    b_dones = np.array([m[4] for m in minibatch])

                    q_current = active_agent.forward(b_states)
                    q_next = active_agent.forward(b_next_states)
                    q_targets = q_current.copy()

                    for b_i in range(batch_size):
                        if b_dones[b_i]:
                            q_targets[b_i, b_actions[b_i]] = b_rewards[b_i]
                        else:
                            q_targets[b_i, b_actions[b_i]] = b_rewards[b_i] + gamma * np.max(q_next[b_i])

                    loss_val = active_agent.train_step(b_states, q_targets)
                    ep_losses.append(loss_val)

            # End of episode for PPO: trajectory update
            if is_ppo:
                pl, vl, ent = active_agent.train_trajectory(
                    ep_states, ep_acts, ep_rews, ep_next_s, ep_dones, ep_lps, ep_vals, gamma=gamma
                )
                history_pol_loss.append(pl)
                history_val_loss.append(vl)
                history_entropy.append(ent)
                history_loss.append(vl)
            elif is_sac:
                q_l_avg = float(np.mean(ep_losses)) if ep_losses else 0.0
                p_l_avg = float(np.mean(ep_pol_losses)) if ep_pol_losses else 0.0
                alp_avg = float(np.mean(ep_alphas)) if ep_alphas else float(active_agent.alpha)
                history_loss.append(q_l_avg)
                history_val_loss.append(q_l_avg)
                history_pol_loss.append(p_l_avg)
                history_entropy.append(alp_avg)
                history_eps.append(alp_avg)
            else:
                eps = max(eps_min, eps * epsilon_decay)
                history_loss.append(np.mean(ep_losses) if ep_losses else 0.0)

            final_nw = float(portfolio.net_worth)
            history_episodes.append(ep + 1)
            history_networth.append(final_nw)
            history_reward.append(ep_reward)
            if not is_sac:
                history_eps.append(eps)

            prog_pct = int(((ep + 1) / n_episodes) * 100)
            status_txt = f"Episode {ep+1}/{n_episodes} | Net Worth: {currency_sym}{final_nw:,.2f} | Reward: {ep_reward:+.3f}"
            if is_sac:
                status_txt += f" | α: {active_agent.alpha:.3f}"
            elif is_ppo:
                status_txt += f" | Ent: {ent:.2f}"
            else:
                status_txt += f" | ε: {eps:.2f}"
            prog_bar.progress(prog_pct, text=status_txt)

        # Out-of-Sample Evaluation
        env_oos, port_oos = create_tt_env(df_test, initial_cash, reward_scheme_choice)
        obs_oos = env_oos.reset()
        st_oos = obs_oos[0].flatten()[:state_dim]
        
        oos_actions = []
        oos_allocations = []
        oos_networths = [initial_cash]
        oos_dates = [df_test.index[0]]
        oos_step = 0
        done_oos = False

        peak_oos_val = initial_cash

        if is_drqn:
            active_agent.reset_hidden()

        while not done_oos and oos_step < len(df_test) - 2:
            curr_nw = float(port_oos.net_worth)
            peak_oos_val = max(peak_oos_val, curr_nw)
            curr_dd = (curr_nw - peak_oos_val) / peak_oos_val

            if is_continuous:
                target_w, _, _ = active_agent.get_action(st_oos, deterministic=True)
                if emergency_kill_switch or curr_dd < -circuit_breaker_mdd:
                    target_w = 0.0
                curr_eq_val = float(port_oos.net_worth) - float(port_oos.wallets[0].balance.as_float())
                curr_w = max(0.0, min(1.0, curr_eq_val / (curr_nw + 1e-6)))
                if target_w - curr_w > 0.05:
                    act_oos = 1 # Buy
                elif curr_w - target_w > 0.05:
                    act_oos = 2 # Sell
                else:
                    act_oos = 0 # Hold
                oos_alloc = target_w * 100.0
            else:
                q_v = active_agent.forward(st_oos)
                raw_act = int(np.argmax(q_v))
                if emergency_kill_switch or curr_dd < -circuit_breaker_mdd:
                    act_oos = 2 # Forced 100% Cash liquidation
                else:
                    act_oos = raw_act
                oos_alloc = 100.0 if act_oos == 1 else (0.0 if act_oos == 2 else 50.0)

            oos_actions.append(act_oos)
            oos_allocations.append(oos_alloc)

            step_res_oos = env_oos.step(act_oos)
            st_oos = step_res_oos[0].flatten()[:state_dim]
            done_oos = bool(step_res_oos[2] or step_res_oos[3])

            oos_networths.append(float(port_oos.net_worth))
            oos_dates.append(df_test.index[min(oos_step + 1, len(df_test)-1)])
            oos_step += 1

        # Format Out-of-Sample Evaluation Results
        oos_eq_series = pd.Series(oos_networths, index=pd.to_datetime(oos_dates))
        oos_bh_series = initial_cash * (df_test["Close"].loc[oos_eq_series.index] / df_test["Close"].loc[oos_eq_series.index[0]])

        # Metrics
        tot_rl_ret = float(((oos_eq_series.iloc[-1] - initial_cash) / initial_cash) * 100.0)
        tot_bh_ret = float(((oos_bh_series.iloc[-1] - initial_cash) / initial_cash) * 100.0)
        n_years = max(len(oos_eq_series) / 252.0, 0.1)
        cagr_rl = float(((oos_eq_series.iloc[-1] / initial_cash) ** (1.0 / n_years) - 1.0) * 100.0)

        peak_eq = np.maximum.accumulate(oos_eq_series)
        dd_rl = (oos_eq_series - peak_eq) / peak_eq
        mdd_rl = float(dd_rl.min()) * 100.0

        daily_rets = oos_eq_series.pct_change().dropna()
        rf_day = 0.05 / 252.0
        ex_rets = daily_rets - rf_day
        shrp_rl = float((np.mean(ex_rets) / (np.std(ex_rets) + 1e-10)) * np.sqrt(252.0))
        down_rets = ex_rets[ex_rets < 0]
        sort_rl = float((np.mean(ex_rets) / (np.std(down_rets) + 1e-10)) * np.sqrt(252.0)) if len(down_rets) > 0 else shrp_rl

        # Save to session state
        st.session_state["rl_trained"] = True
        st.session_state["rl_q_net"] = active_agent
        st.session_state["rl_agent_type"] = agent_arch
        st.session_state["rl_history"] = {
            "episodes": history_episodes,
            "networth": history_networth,
            "reward": history_reward,
            "loss": history_loss,
            "pol_loss": history_pol_loss,
            "val_loss": history_val_loss,
            "entropy": history_entropy,
            "epsilon": history_eps
        }
        st.session_state["rl_oos_res"] = {
            "equity": oos_eq_series,
            "bh_equity": oos_bh_series,
            "actions": oos_actions,
            "allocations": oos_allocations,
            "tot_ret": tot_rl_ret,
            "tot_bh": tot_bh_ret,
            "cagr": cagr_rl,
            "mdd": mdd_rl,
            "sharpe": shrp_rl,
            "sortino": sort_rl,
            "alpha": tot_rl_ret - tot_bh_ret
        }
        st.success(f"✅ Training completed! Out-of-Sample Return: `{tot_rl_ret:+.2f}%` vs Buy & Hold: `{tot_bh_ret:+.2f}%`.")

# ---------------------------------------------------------
# Top Highlights Banner (if trained)
# ---------------------------------------------------------
if st.session_state["rl_trained"] and st.session_state["rl_oos_res"] is not None:
    res_top = st.session_state["rl_oos_res"]
    t1, t2, t3, t4, t5 = st.columns(5)
    t1.metric("Out-of-Sample Return", f"{res_top['tot_ret']:+.2f}%", delta=f"{res_top['alpha']:+.2f}% Alpha vs B&H")
    t2.metric("Annualized CAGR", f"{res_top['cagr']:+.2f}%")
    t3.metric("Annualized Sharpe", f"{res_top['sharpe']:.2f}")
    t4.metric("Annualized Sortino", f"{res_top['sortino']:.2f}")
    t5.metric("Max Drawdown", f"{res_top['mdd']:.2f}%")

    # Current Live Bar Action Recommendation
    if st.session_state["rl_q_net"] is not None:
        curr_feat = np.array([
            df_feat["Price_Norm"].iloc[-1],
            df_feat["Return"].iloc[-1],
            df_feat["SMA_Ratio"].iloc[-1],
            df_feat["RSI_Norm"].iloc[-1],
            df_feat["MACD"].iloc[-1],
            df_feat["Vol_Ratio"].iloc[-1]
        ])
        is_ppo_active = "PPO" in st.session_state.get("rl_agent_type", agent_arch)
        is_sac_active = "SAC" in st.session_state.get("rl_agent_type", agent_arch)

        c_sig1, c_sig2 = st.columns([3, 2])
        if is_sac_active:
            live_w, live_lp, live_v = st.session_state["rl_q_net"].get_action(curr_feat, deterministic=True)
            w_pct = live_w * 100.0
            q1_val, _, _, _ = st.session_state["rl_q_net"].forward_q(
                st.session_state["rl_q_net"].w1_q1, st.session_state["rl_q_net"].b1_q1,
                st.session_state["rl_q_net"].w2_q1, st.session_state["rl_q_net"].b2_q1,
                st.session_state["rl_q_net"].w3_q1, st.session_state["rl_q_net"].b3_q1,
                curr_feat, np.array([[live_w]])
            )
            q2_val, _, _, _ = st.session_state["rl_q_net"].forward_q(
                st.session_state["rl_q_net"].w1_q2, st.session_state["rl_q_net"].b1_q2,
                st.session_state["rl_q_net"].w2_q2, st.session_state["rl_q_net"].b2_q2,
                st.session_state["rl_q_net"].w3_q2, st.session_state["rl_q_net"].b3_q2,
                curr_feat, np.array([[live_w]])
            )
            q1_f, q2_f = float(q1_val[0, 0]), float(q2_val[0, 0])
            disagree = abs(q1_f - q2_f)
            alpha_curr = float(st.session_state["rl_q_net"].alpha)
            with c_sig1:
                st.info(f"⚡ **Real-Time SAC Maximum Entropy Allocation (Latest Bar):** **{w_pct:.1f}% Equity / {100.0 - w_pct:.1f}% Cash** | Twin Q1: `{currency_sym}{q1_f:,.2f}` | Q2: `{currency_sym}{q2_f:,.2f}` | Status: {'🚨 LIQUIDATED BY KILL-SWITCH' if emergency_kill_switch else '🟢 ACTIVE'}")
            with c_sig2:
                order_hint = "BUY / ACCUMULATE" if live_w > 0.55 else ("SELL / REDUCE" if live_w < 0.45 else "HOLD / REBALANCE")
                st.caption(rf"**SAC Entropy Temp:** $\alpha={alpha_curr:.3f}$ | Critic Disagreement: $\Delta Q={disagree:.3f}$ | Signal: **{order_hint}**")
        elif is_ppo_active:
            live_w, _, live_v = st.session_state["rl_q_net"].get_action(curr_feat, deterministic=True)
            w_pct = live_w * 100.0
            with c_sig1:
                st.info(f"⚡ **Real-Time PPO Target Allocation (Latest Bar):** **{w_pct:.1f}% Equity / {100.0 - w_pct:.1f}% Cash** | State Value: `{currency_sym}{live_v:,.2f}` | Status: {'🚨 LIQUIDATED BY KILL-SWITCH' if emergency_kill_switch else '🟢 ACTIVE'}")
            with c_sig2:
                order_hint = "BUY / ACCUMULATE" if live_w > 0.55 else ("SELL / REDUCE" if live_w < 0.45 else "HOLD / REBALANCE")
                st.caption(rf"**Policy Distribution:** $\mu={live_w:.3f}$ | Signal: **{order_hint}**")
        else:
            live_q = st.session_state["rl_q_net"].forward(curr_feat).flatten()
            live_act = int(np.argmax(live_q))
            act_labels = {0: "HOLD (Cash / Neutral)", 1: "BUY (Long Allocation)", 2: "SELL / FLAT (Stay in Cash)"}
            q_conf = float(np.max(live_q) - np.sort(live_q)[-2])
            with c_sig1:
                st.info(f"⚡ **Real-Time RL Policy Signal (Latest Bar):** **{act_labels[live_act]}** | Confidence Margin: `+{q_conf:.4f}` | Status: {'🚨 LIQUIDATED BY KILL-SWITCH' if emergency_kill_switch else '🟢 ACTIVE'}")
            with c_sig2:
                st.caption(f"**Q-Values on Current State:** Q(Hold)=`{live_q[0]:.3f}` | Q(Buy)=`{live_q[1]:.3f}` | Q(Sell)=`{live_q[2]:.3f}`")

    st.markdown("---")

# ---------------------------------------------------------
# 8-Tab Modular Reinforcement Learning Suite
# ---------------------------------------------------------
tab_work, tab_oos, tab_actions, tab_qval, tab_rewards, tab_tourn, tab_method, tab_exports = st.tabs([
    "🎮 Training Telemetry & Workbench",
    "📈 Out-of-Sample Equity & Drawdown",
    "🎯 Policy Execution Markers",
    "🧠 Q-Value Surface Inspection",
    "🔬 Reward Engineering Comparison",
    "🏆 RL vs Quantitative Tournament",
    "📚 Methodology & TensorTrade Architecture",
    "📋 Order Log & CSV Tearsheets"
])

# =========================================================
# TAB 1: Training Telemetry & Workbench
# =========================================================
with tab_work:
    st.subheader("🎮 Training Telemetry & Learning Dynamics")
    st.caption("Inspect how the neural agent explores state space, minimizes loss, and optimizes portfolio value.")

    if st.session_state["rl_trained"] and st.session_state["rl_history"] is not None:
        hist = st.session_state["rl_history"]
        is_ppo_hist = "PPO" in st.session_state.get("rl_agent_type", "")
        is_sac_hist = "SAC" in st.session_state.get("rl_agent_type", "")
        
        c_tr1, c_tr2 = st.columns(2)
        with c_tr1:
            st.markdown("#### Episode Net Worth Progression")
            fig_nw = go.Figure()
            fig_nw.add_trace(go.Scatter(x=hist["episodes"], y=hist["networth"], mode="lines+markers", line=dict(color="#00E676", width=2.0), name="Terminal Net Worth"))
            fig_nw.add_hline(y=initial_cash, line_dash="dash", line_color="#38BDF8", annotation_text="Initial Capital")
            fig_nw.update_layout(template="plotly_dark", height=320, xaxis=dict(title="Training Episode"), yaxis=dict(title=f"Net Worth ({currency_sym})"))
            st.plotly_chart(fig_nw, width="stretch")

        with c_tr2:
            st.markdown("#### Cumulative Reward & Policy Regularization")
            fig_rw = make_subplots(specs=[[{"secondary_y": True}]])
            fig_rw.add_trace(go.Scatter(x=hist["episodes"], y=hist["reward"], mode="lines+markers", line=dict(color="#F59E0B", width=2.0), name="Total Reward"), secondary_y=False)
            if is_sac_hist:
                fig_rw.add_trace(go.Scatter(x=hist["episodes"], y=hist["entropy"], mode="lines", line=dict(color="#38BDF8", width=2.0, dash="solid"), name="Adaptive Temp α"), secondary_y=True)
                fig_rw.update_yaxes(title_text="Temperature (α)", secondary_y=True)
            elif is_ppo_hist and "entropy" in hist and len(hist["entropy"]) > 0:
                fig_rw.add_trace(go.Scatter(x=hist["episodes"], y=hist["entropy"], mode="lines", line=dict(color="#38BDF8", width=1.5, dash="dot"), name="Policy Entropy H(π)"), secondary_y=True)
                fig_rw.update_yaxes(title_text="Policy Entropy", secondary_y=True)
            else:
                fig_rw.add_trace(go.Scatter(x=hist["episodes"], y=hist["epsilon"], mode="lines", line=dict(color="#94A3B8", width=1.5, dash="dot"), name="Epsilon (ε)"), secondary_y=True)
                fig_rw.update_yaxes(title_text="Exploration Rate (ε)", secondary_y=True)
            fig_rw.update_layout(template="plotly_dark", height=320, xaxis=dict(title="Training Episode"))
            fig_rw.update_yaxes(title_text="Episode Cumulative Reward", secondary_y=False)
            st.plotly_chart(fig_rw, width="stretch")

        if is_sac_hist and "pol_loss" in hist and len(hist["pol_loss"]) > 0:
            st.markdown("#### SAC Twin Critic & Policy Loss Decomposition")
            c_loss1, c_loss2 = st.columns(2)
            with c_loss1:
                fig_vloss = go.Figure()
                fig_vloss.add_trace(go.Scatter(x=hist["episodes"], y=hist["loss"], mode="lines", line=dict(color="#FF5252", width=1.8), name="Twin Critic MSE Loss"))
                fig_vloss.update_layout(template="plotly_dark", height=240, xaxis=dict(title="Episode"), yaxis=dict(title="Critic Loss"))
                st.plotly_chart(fig_vloss, width="stretch")
            with c_loss2:
                fig_ploss = go.Figure()
                fig_ploss.add_trace(go.Scatter(x=hist["episodes"], y=hist["pol_loss"], mode="lines", line=dict(color="#38BDF8", width=1.8), name="Actor Policy Loss"))
                fig_ploss.update_layout(template="plotly_dark", height=240, xaxis=dict(title="Episode"), yaxis=dict(title="Policy Loss"))
                st.plotly_chart(fig_ploss, width="stretch")
        elif is_ppo_hist and "pol_loss" in hist and len(hist["pol_loss"]) > 0:
            st.markdown("#### PPO Actor-Critic Loss Decomposition")
            c_loss1, c_loss2 = st.columns(2)
            with c_loss1:
                fig_ploss = go.Figure()
                fig_ploss.add_trace(go.Scatter(x=hist["episodes"], y=hist["pol_loss"], mode="lines", line=dict(color="#38BDF8", width=1.8), name="Surrogate Policy Loss"))
                fig_ploss.update_layout(template="plotly_dark", height=240, xaxis=dict(title="Episode"), yaxis=dict(title="Policy Loss"))
                st.plotly_chart(fig_ploss, width="stretch")
            with c_loss2:
                fig_vloss = go.Figure()
                fig_vloss.add_trace(go.Scatter(x=hist["episodes"], y=hist["val_loss"], mode="lines", line=dict(color="#FF5252", width=1.8), name="Critic Value MSE Loss"))
                fig_vloss.update_layout(template="plotly_dark", height=240, xaxis=dict(title="Episode"), yaxis=dict(title="Critic Loss"))
                st.plotly_chart(fig_vloss, width="stretch")
        else:
            st.markdown("#### TD Loss Convergence Curve")
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(x=hist["episodes"], y=hist["loss"], mode="lines", line=dict(color="#FF5252", width=1.8), name="MSE Bellman Loss"))
            fig_loss.update_layout(template="plotly_dark", height=260, xaxis=dict(title="Episode"), yaxis=dict(title="TD Loss"))
            st.plotly_chart(fig_loss, width="stretch")
    else:
        st.info("Click **'▶ Train Reinforcement Learning Agent'** above to initiate training and generate telemetry curves.")

# =========================================================
# TAB 2: Out-of-Sample Equity & Drawdown
# =========================================================
with tab_oos:
    st.subheader("📈 Out-of-Sample Validation & Equity Profile")
    st.caption("Strict out-of-sample evaluation on unseen market bars where the trained neural agent acts deterministically (zero exploration).")

    if st.session_state["rl_trained"] and st.session_state["rl_oos_res"] is not None:
        oos = st.session_state["rl_oos_res"]
        active_type = st.session_state.get("rl_agent_type", agent_arch)

        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=oos["equity"].index, y=oos["equity"].values, mode="lines", name=f"RL Policy ({active_type})", line=dict(color="#00E676", width=2.4)))
        fig_eq.add_trace(go.Scatter(x=oos["bh_equity"].index, y=oos["bh_equity"].values, mode="lines", name=f"Buy & Hold {company}", line=dict(color="#38BDF8", width=1.6, dash="dash")))
        fig_eq.update_layout(template="plotly_dark", height=400, title="Cumulative Portfolio Equity: RL Policy vs Buy & Hold", yaxis=dict(title=f"Equity ({currency_sym})"))
        st.plotly_chart(fig_eq, width="stretch")

        # Continuous Position Allocation Band
        if "allocations" in oos and len(oos["allocations"]) > 0:
            st.markdown("#### 🎯 Dynamic Portfolio Allocation Exposure Band (% Target Weight)")
            p_dates_alloc = oos["equity"].index[:len(oos["allocations"])]
            fig_band = go.Figure()
            fig_band.add_trace(go.Scatter(
                x=p_dates_alloc,
                y=oos["allocations"],
                mode="lines",
                line=dict(color="#00E676", width=2.0),
                fill="tozeroy",
                fillcolor="rgba(0, 230, 118, 0.22)",
                name="Stock Target Weight (%)"
            ))
            fig_band.add_hline(y=100.0, line_dash="dot", line_color="#94A3B8", annotation_text="100% Equity")
            fig_band.add_hline(y=0.0, line_dash="dot", line_color="#FF5252", annotation_text="100% Cash")
            fig_band.update_layout(template="plotly_dark", height=240, yaxis=dict(title="Exposure %", range=[-5, 110]))
            st.plotly_chart(fig_band, width="stretch")

        # Underwater Drawdown Plot
        c_eq_pk = np.maximum.accumulate(oos["equity"])
        c_dd = ((oos["equity"] - c_eq_pk) / c_eq_pk) * 100.0

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=oos["equity"].index, y=c_dd.values, mode="lines", fill="tozeroy", fillcolor="rgba(244, 63, 94, 0.25)", line=dict(color="#FF5252", width=1.5), name="RL Drawdown %"))
        fig_dd.update_layout(template="plotly_dark", height=260, title="Underwater Drawdown Profile (%)", yaxis=dict(title="Drawdown %"))
        st.plotly_chart(fig_dd, width="stretch")
    else:
        st.info("Agent has not yet been trained. Launch training above to view out-of-sample performance.")

# =========================================================
# TAB 3: Policy Execution Markers
# =========================================================
with tab_actions:
    st.subheader("🎯 Policy Action Execution Markers")
    st.caption("Inspect individual order signals generated along the historical price curve.")

    if st.session_state["rl_trained"] and st.session_state["rl_oos_res"] is not None:
        oos = st.session_state["rl_oos_res"]
        act_arr = oos["actions"]
        p_dates = oos["equity"].index[:len(act_arr)]
        p_closes = df_test["Close"].loc[p_dates]

        # Buy and Sell markers
        buy_mask = np.array(act_arr) == 1
        sell_mask = np.array(act_arr) == 2

        fig_act = go.Figure()
        fig_act.add_trace(go.Scatter(x=p_dates, y=p_closes.values, mode="lines", line=dict(color="#94A3B8", width=1.4), name="Close Price"))
        
        if np.any(buy_mask):
            fig_act.add_trace(go.Scatter(x=p_dates[buy_mask], y=p_closes.values[buy_mask], mode="markers", marker=dict(symbol="triangle-up", color="#00E676", size=9), name="Buy Order"))
        if np.any(sell_mask):
            fig_act.add_trace(go.Scatter(x=p_dates[sell_mask], y=p_closes.values[sell_mask], mode="markers", marker=dict(symbol="triangle-down", color="#FF5252", size=9), name="Sell / Flat Order"))

        fig_act.update_layout(template="plotly_dark", height=420, title="Executed Orders Along Out-of-Sample Price Path", yaxis=dict(title=f"Price ({currency_sym})"))
        st.plotly_chart(fig_act, width="stretch")

        # Action Distribution
        c_n_buy = int(np.sum(buy_mask))
        c_n_sell = int(np.sum(sell_mask))
        c_n_hold = len(act_arr) - c_n_buy - c_n_sell
        
        df_dist = pd.DataFrame([
            {"Action": "Buy (Long)", "Count": c_n_buy, "Percentage": f"{c_n_buy/len(act_arr)*100:.1f}%"},
            {"Action": "Sell / Flat", "Count": c_n_sell, "Percentage": f"{c_n_sell/len(act_arr)*100:.1f}%"},
            {"Action": "Hold (Neutral)", "Count": c_n_hold, "Percentage": f"{c_n_hold/len(act_arr)*100:.1f}%"}
        ])

        c_ad1, c_ad2 = st.columns(2)
        with c_ad1:
            st.markdown("#### Action Distribution Summary")
            st.dataframe(df_dist, width="stretch", hide_index=True)
        with c_ad2:
            fig_dist_bar = px.bar(df_dist, x="Action", y="Count", color="Action", color_discrete_sequence=["#00E676", "#FF5252", "#94A3B8"])
            fig_dist_bar.update_layout(template="plotly_dark", height=280)
            st.plotly_chart(fig_dist_bar, width="stretch")
    else:
        st.info("Execute agent training to view policy execution markers.")

# =========================================================
# TAB 4: Q-Value & Policy Surface Inspection
# =========================================================
with tab_qval:
    st.subheader("🧠 State-Action Value & Policy Decision Surface")
    st.caption("Inspect what action or continuous exposure the trained agent selects across varying RSI and price momentum states.")

    if st.session_state["rl_trained"] and st.session_state["rl_q_net"] is not None:
        agent_inst = st.session_state["rl_q_net"]
        agent_type_str = st.session_state.get("rl_agent_type", "")
        is_ppo_active = "PPO" in agent_type_str
        is_sac_active = "SAC" in agent_type_str
        
        # Build 2D grid: RSI (-1 to +1) x SMA_Ratio (-0.05 to +0.05)
        rsi_grid = np.linspace(-1.0, 1.0, 20)
        sma_grid = np.linspace(-0.05, 0.05, 20)
        surface_vals = np.zeros((len(sma_grid), len(rsi_grid)))
        uncert_vals = np.zeros((len(sma_grid), len(rsi_grid)))

        for i_s, s_val in enumerate(sma_grid):
            for j_r, r_val in enumerate(rsi_grid):
                # Synthetic state: [price_norm, return, sma_ratio, rsi_norm, macd, vol_ratio]
                synth_state = np.array([0.0, 0.0, s_val, r_val, s_val*0.5, 0.0])
                if is_sac_active:
                    w, _, _ = agent_inst.get_action(synth_state, deterministic=True)
                    surface_vals[i_s, j_r] = w * 100.0
                    q1_s, _, _, _ = agent_inst.forward_q(
                        agent_inst.w1_q1, agent_inst.b1_q1, agent_inst.w2_q1, agent_inst.b2_q1, agent_inst.w3_q1, agent_inst.b3_q1,
                        synth_state, np.array([[w]])
                    )
                    q2_s, _, _, _ = agent_inst.forward_q(
                        agent_inst.w1_q2, agent_inst.b1_q2, agent_inst.w2_q2, agent_inst.b2_q2, agent_inst.w3_q2, agent_inst.b3_q2,
                        synth_state, np.array([[w]])
                    )
                    uncert_vals[i_s, j_r] = abs(float(q1_s[0, 0]) - float(q2_s[0, 0]))
                elif is_ppo_active:
                    w, _, _ = agent_inst.get_action(synth_state, deterministic=True)
                    surface_vals[i_s, j_r] = w * 100.0
                else:
                    q_vals = agent_inst.forward(synth_state).flatten()
                    surface_vals[i_s, j_r] = q_vals[1] - q_vals[2]

        if is_sac_active:
            c_sf1, c_sf2 = st.columns(2)
            with c_sf1:
                fig_surf = px.imshow(
                    surface_vals,
                    x=[f"{r:.1f}" for r in rsi_grid],
                    y=[f"{s*100:.1f}%" for s in sma_grid],
                    labels=dict(x="RSI State (Normalized -1 to +1)", y="SMA Ratio (Trend Momentum)", color="Target Equity %"),
                    color_continuous_scale="Viridis",
                    title="SAC Continuous Allocation Surface (% Target)"
                )
                st.plotly_chart(fig_surf, width="stretch")
                st.caption("💡 **Interpretation:** Yellow/green shows states where SAC policy allocates aggressively to equities; dark purple de-risks into cash.")
            with c_sf2:
                fig_uncert = px.imshow(
                    uncert_vals,
                    x=[f"{r:.1f}" for r in rsi_grid],
                    y=[f"{s*100:.1f}%" for s in sma_grid],
                    labels=dict(x="RSI State (Normalized -1 to +1)", y="SMA Ratio (Trend Momentum)", color="|Q1 - Q2| Disagreement"),
                    color_continuous_scale="Plasma",
                    title="Twin Critic Disagreement |Q1 - Q2| (Epistemic Uncertainty)"
                )
                st.plotly_chart(fig_uncert, width="stretch")
                st.caption("💡 **Interpretation:** Bright peaks indicate regions of state space where the twin critics disagree most, highlighting market regimes with elevated model uncertainty.")
        elif is_ppo_active:
            fig_surf = px.imshow(
                surface_vals,
                x=[f"{r:.1f}" for r in rsi_grid],
                y=[f"{s*100:.1f}%" for s in sma_grid],
                labels=dict(x="RSI State (Normalized -1 to +1)", y="SMA Ratio (Trend Momentum)", color="Target Equity %"),
                color_continuous_scale="Viridis",
                title="PPO Continuous Policy Allocation Surface (% Equity Target)"
            )
            st.plotly_chart(fig_surf, width="stretch")
            st.caption("💡 **Interpretation:** Bright yellow/green regions show states where PPO targets high equity allocation (~100%). Dark purple regions show where the policy de-risks into cash (~0%).")
        else:
            fig_surf = px.imshow(
                surface_vals,
                x=[f"{r:.1f}" for r in rsi_grid],
                y=[f"{s*100:.1f}%" for s in sma_grid],
                labels=dict(x="RSI State (Normalized -1 to +1)", y="SMA Ratio (Trend Momentum)", color="Q(Buy) - Q(Sell)"),
                color_continuous_scale="RdYlGn",
                title="Q-Network Preference: Q(Buy) - Q(Sell) Advantage Spread"
            )
            st.plotly_chart(fig_surf, width="stretch")
            st.caption("💡 **Interpretation:** Green regions indicate states where the Q-network prefers taking a Long position. Red regions indicate states where the agent favors selling or staying in cash.")
    else:
        st.info("Train the RL agent above to inspect its learned decision boundaries.")

# =========================================================
# TAB 5: Reward Engineering Comparison
# =========================================================
with tab_rewards:
    st.subheader("🔬 Reward Engineering & Scheme Comparison")
    st.caption("How objective reward formulations influence agent risk aversion and portfolio drawdown.")

    st.markdown(r"""
    The choice of reward function dictates whether the agent develops aggressive momentum seeking or conservative capital preservation behavior:

    1. **Simple Profit Scheme ($R_{\text{Profit}}$):**
       $$R_t = \frac{W_t - W_{t-1}}{W_{t-1}}$$
       Directly incentivizes maximum portfolio return without penalizing downside volatility. Tends to hold volatile assets through severe drawdowns.

    2. **Risk-Adjusted Sortino Scheme ($R_{\text{Risk}}$):**
       $$R_t = \frac{\Delta W_t}{W_{t-1}} - \lambda \cdot \max\left(0, -\frac{\Delta W_t}{W_{t-1}}\right)^2$$
       Asymmetrically penalizes downside drawdowns. The agent learns to trim positions before high-volatility cascade selloffs.

    3. **Position-Based Return Scheme ($R_{\text{PBR}}$):**
       $$R_t = a_t \cdot r_t - c_{\text{comm}} \cdot |a_t - a_{t-1}|$$
       Directly measures alignment between position exposure $a_t \in [-1, +1]$ and asset return $r_t$, while explicitly docking transaction frictions.
    """)

    # Reward metric comparison table
    df_reward_comp = pd.DataFrame([
        {"Reward Formulation": "Simple Profit", "Optimizes For": "Cumulative Wealth", "Drawdown Penalty": "None", "Recommended When": "Bullish Trending Regimes"},
        {"Reward Formulation": "Risk-Adjusted (Sortino)", "Optimizes For": "Sharpe / Sortino Ratio", "Drawdown Penalty": "Quadratic Downside Penalty", "Recommended When": "Volatile / Uncertain Regimes"},
        {"Reward Formulation": "Position-Based (PBR)", "Optimizes For": "Action Timing & Low Turnover", "Drawdown Penalty": "Friction & Turnover Deduction", "Recommended When": "High-Frequency Execution"}
    ])
    st.dataframe(df_reward_comp, width="stretch", hide_index=True)

# =========================================================
# TAB 6: RL vs Quantitative Tournament
# =========================================================
with tab_tourn:
    st.subheader("🏆 RL vs Quantitative Strategy Tournament")
    st.caption("Cross-sectional tournament comparing the trained RL agent against classical systematic rules over the identical out-of-sample test window.")

    if st.session_state["rl_trained"] and st.session_state["rl_oos_res"] is not None:
        oos = st.session_state["rl_oos_res"]
        c_test = df_test["Close"].values
        N_tb = len(c_test)

        # Baseline 1: Buy & Hold
        bh_cagr = float(((c_test[-1] / c_test[0]) ** (1.0 / max(N_tb/252.0, 0.1)) - 1.0) * 100.0)
        bh_ret = float(((c_test[-1] - c_test[0]) / c_test[0]) * 100.0)

        # Baseline 2: SMA Crossover (10 / 30)
        s10 = df_test["SMA_10"].values
        s30 = df_test["SMA_30"].values
        sma_sig = (s10 > s30).astype(float)
        sma_rets = sma_sig[:-1] * np.diff(c_test) / c_test[:-1]
        sma_ret = float(((np.prod(1.0 + sma_rets) - 1.0)) * 100.0)

        # Baseline 3: RSI Mean Reversion
        rsi_vals = df_test["RSI"].values
        rsi_sig = (rsi_vals < 40).astype(float)
        rsi_rets = rsi_sig[:-1] * np.diff(c_test) / c_test[:-1]
        rsi_ret = float(((np.prod(1.0 + rsi_rets) - 1.0)) * 100.0)

        tourn_matrix = [
            {"Strategy": "🤖 RL Agent (TensorTrade Policy)", "Total Return (%)": oos["tot_ret"], "CAGR (%)": f"{oos['cagr']:+.2f}%", "Sharpe": f"{oos['sharpe']:.2f}", "Max DD (%)": f"{oos['mdd']:.2f}%", "Type": "Neural RL"},
            {"Strategy": "📈 Buy & Hold Benchmark", "Total Return (%)": bh_ret, "CAGR (%)": f"{bh_cagr:+.2f}%", "Sharpe": "—", "Max DD (%)": "—", "Type": "Passive"},
            {"Strategy": "⚡ SMA Crossover (10/30)", "Total Return (%)": sma_ret, "CAGR (%)": "—", "Sharpe": "—", "Max DD (%)": "—", "Type": "Technical"},
            {"Strategy": "🔄 RSI Mean-Reversion", "Total Return (%)": rsi_ret, "CAGR (%)": "—", "Sharpe": "—", "Max DD (%)": "—", "Type": "Technical"}
        ]

        df_tm = pd.DataFrame(tourn_matrix).sort_values("Total Return (%)", ascending=False).reset_index(drop=True)
        df_tm_disp = df_tm.copy()
        df_tm_disp["Total Return (%)"] = df_tm_disp["Total Return (%)"].apply(lambda x: f"{x:+.2f}%")
        st.dataframe(df_tm_disp, width="stretch", hide_index=True)

        fig_tm_bar = px.bar(
            df_tm, x="Total Return (%)", y="Strategy", orientation="h",
            color="Total Return (%)", color_continuous_scale="Viridis",
            title="Out-of-Sample Performance Comparison"
        )
        fig_tm_bar.update_layout(template="plotly_dark", height=320)
        st.plotly_chart(fig_tm_bar, width="stretch")
    else:
        st.info("Train the RL agent to populate tournament benchmarking.")

# =========================================================
# TAB 7: Reinforcement Learning & TensorTrade Methodology
# =========================================================
with tab_method:
    st.subheader("📚 Reinforcement Learning & TensorTrade Methodology")

    with st.expander("📖 Markov Decision Process (MDP) in Financial Markets", expanded=True):
        st.markdown(r"""
        > [!NOTE]
        > Algorithmic trading is formalized as a continuous-state, discrete-action **Markov Decision Process**:
        > $$\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$$
        >
        > - **State Space $\mathcal{S} \in \mathbb{R}^6$:** Price relative to moving average, log return, SMA trend spread, normalized RSI, MACD signal, and volume ratio.
        > - **Action Space $\mathcal{A} = \{0, 1, 2\}$:** `0: Hold/Cash`, `1: Buy (Long Allocation)`, `2: Sell (Flat/Short)`.
        > - **Transition Probability $\mathcal{P}(s_{t+1} | s_t, a_t)$:** Market price dynamics governed by stochastic asset order flow.
        > - **Reward $\mathcal{R}(s_t, a_t)$:** Portfolio value change minus transaction execution slippage.
        > - **Discount Factor $\gamma \in [0, 1)$:** Balances immediate profits vs. long-term portfolio growth.
        """)

    with st.expander("📖 Deep Q-Learning & Bellman Optimality Equation", expanded=False):
        st.markdown(r"""
        The optimal action-value function $Q^*(s, a)$ satisfies the **Bellman Optimality Equation**:
        $$Q^*(s, a) = \mathbb{E}\left[ R_{t+1} + \gamma \max_{a'} Q^*(S_{t+1}, a') \;\middle|\; S_t = s, A_t = a \right]$$

        The Q-network parameters $\theta$ are updated via gradient descent to minimize Mean Squared Bellman Error:
        $$L(\theta) = \mathbb{E}_{\mathcal{D}}\left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]$$
        where $\theta^-$ denotes target network weights periodically synced to stabilize training.
        """)

    with st.expander("📖 Proximal Policy Optimization (PPO) & Generalized Advantage Estimation (GAE)", expanded=False):
        st.markdown(r"""
        **Proximal Policy Optimization (PPO)** optimizes continuous target asset weights $w_t \in [0.0, 1.0]$ via a dual-head Actor-Critic network with clipped surrogate objectives:

        1. **Probability Ratio**:
           $$r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$$

        2. **Clipped Surrogate Loss Objective**:
           $$\mathcal{L}^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta)\hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]$$
           where $\epsilon \in [0.1, 0.3]$ clips excessively large policy steps to prevent policy collapse.

        3. **Generalized Advantage Estimation (GAE-$\lambda$):**
           $$\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$$
           $$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}^V$$
           GAE smoothly interpolates between high-variance Monte Carlo returns ($\lambda=1$) and high-bias 1-step TD errors ($\lambda=0$).

        4. **Joint Loss Formulation**:
           $$\mathcal{L}^{\text{Total}} = -\mathcal{L}^{\text{CLIP}}(\theta) + c_1 \cdot \text{MSE}\left(V_\phi(s_t), R_t^{\text{target}}\right) - c_2 \cdot \mathcal{H}\left(\pi_\theta(\cdot | s_t)\right)$$
           where $\mathcal{H}(\pi)$ enforces policy entropy to ensure continuous exploration.
        """)

    with st.expander("📖 Soft Actor-Critic (SAC): Maximum Entropy RL & Twin Critic Bellman Updates", expanded=False):
        st.markdown(r"""
        **Soft Actor-Critic (SAC)** is an off-policy actor-critic algorithm that optimizes continuous portfolio exposure $w_t \in [0.0, 1.0]$ under the **Maximum Entropy Reinforcement Learning** framework:

        1. **Maximum Entropy Objective**:
           $$J(\pi) = \sum_{t=0}^T \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} \left[ R(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot | s_t)) \right]$$
           where the policy $\pi$ is incentivized to maximize expected cumulative returns while maximizing action entropy $\mathcal{H}(\pi) = -\mathbb{E}[\log \pi(a|s)]$, preventing premature collapse onto deterministic sub-optimal trading strategies.

        2. **Twin Critics ($Q_1, Q_2$) & Target Bellman Equation**:
           To counteract overestimation bias in high-volatility financial regimes, SAC maintains two independent critic networks and evaluates the minimum target:
           $$y_t = r_t + \gamma (1 - d_t) \left( \min_{j \in \{1, 2\}} Q_{\text{target}, j}(s_{t+1}, \tilde{a}_{t+1}) - \alpha \log \pi(\tilde{a}_{t+1} | s_{t+1}) \right)$$
           $$\mathcal{L}(Q_j) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}} \left[ \frac{1}{2} \left( Q_j(s, a) - y_t \right)^2 \right]$$

        3. **Reparameterized Sigmoidal Continuous Policy**:
           The actor outputs Gaussian parameters $[\mu(s), \log \sigma(s)]$. A latent sample $u \sim \mathcal{N}(\mu, \sigma^2)$ is mapped into valid portfolio weight space via a smooth sigmoid squash:
           $$a = \sigma(u) = \frac{1}{1 + e^{-u}} \in (0, 1)$$
           With exact analytical Jacobian correction for bounded density evaluation:
           $$\log \pi(a | s) = \log \mathcal{N}(u; \mu, \sigma^2) - \log\left(a(1 - a) + 10^{-7}\right)$$

        4. **Automatic Temperature Tuning ($\alpha$)**:
           The temperature parameter $\alpha$ dynamically balances exploration vs. exploitation:
           $$\mathcal{L}(\alpha) = \mathbb{E}_{a \sim \pi} \left[ -\alpha \left( \log \pi(a | s) + \bar{\mathcal{H}} \right) \right]$$
           where target entropy is set to $\bar{\mathcal{H}} = -\text{dim}(\mathcal{A}) = -1.0$.

        5. **Polyak Target Smoothing Updates**:
           Target critic weights $\theta_{\text{target}}$ are smoothly updated after each gradient step without discontinuous target shocks:
           $$\theta_{\text{target}} \leftarrow \tau \theta + (1 - \tau) \theta_{\text{target}}, \quad \tau \in [0.01, 0.20]$$
        """)

    with st.expander("📖 TensorTrade Component Lifecycle Architecture", expanded=False):
        st.markdown(r"""
        TensorTrade decouples quantitative trading into clean, modular building blocks:
        1. **`Instrument`**: Defines tradable assets (`USD`, `INR`, `STOCK`).
        2. **`Exchange`**: Simulates order matching, latency, and fills (`execute_order`).
        3. **`Wallet` & `Portfolio`**: Manages cash balances, equity allocations, and net worth accounting.
        4. **`DataFeed`**: Lazily evaluates real-time feature streams (`Stream.source(...)`).
        5. **`ActionScheme`**: Maps neural network actions into broker orders (`BSH`, `SimpleOrders`).
        6. **`RewardScheme`**: Translates portfolio transitions into numerical learning signals (`SimpleProfit`, `RiskAdjustedReturns`).
        """)

# =========================================================
# TAB 8: Order Log & CSV Tearsheets
# =========================================================
with tab_exports:
    st.subheader("📋 Order Execution Log & CSV Tearsheets")

    if st.session_state["rl_trained"] and st.session_state["rl_oos_res"] is not None:
        oos = st.session_state["rl_oos_res"]
        act_arr = oos["actions"]
        p_dates = oos["equity"].index[:len(act_arr)]
        p_closes = df_test["Close"].loc[p_dates].values

        action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}
        df_order_log = pd.DataFrame({
            "Date": [d.strftime("%Y-%m-%d") for d in p_dates],
            "Close_Price": [f"{currency_sym}{p:,.2f}" for p in p_closes],
            "Policy_Action": [action_names.get(a, "HOLD") for a in act_arr],
            "Portfolio_Net_Worth": [f"{currency_sym}{nw:,.2f}" for nw in oos["equity"].values[:len(act_arr)]]
        })

        st.dataframe(df_order_log, width="stretch", hide_index=True)

        c_dl1, c_dl2 = st.columns(2)
        with c_dl1:
            st.download_button(
                label=f"📥 Download RL Order Execution Log CSV ({ticker})",
                data=df_order_log.to_csv(index=False).encode("utf-8"),
                file_name=f"rl_orders_{ticker}_{datetime.date.today().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                width="stretch",
                key="rl_btn_dl_orders_csv"
            )
        with c_dl2:
            export_oos_df = pd.DataFrame({
                "Date": [d.strftime("%Y-%m-%d") for d in oos["equity"].index],
                "RL_Policy_Equity": oos["equity"].values,
                "Buy_Hold_Equity": oos["bh_equity"].values
            })
            st.download_button(
                label="📥 Export Out-of-Sample Equity Curve (CSV)",
                data=export_oos_df.to_csv(index=False).encode("utf-8"),
                file_name=f"rl_equity_{ticker}_{datetime.date.today().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                width="stretch",
                key="rl_btn_dl_equity_csv"
            )
    else:
        st.info("Train the agent to generate and export order execution logs.")

st.markdown("---")
st.markdown("<div style='text-align: center; margin-top: 15px; color: #64748B; font-size: 0.78rem;'><i>QuantTerminal RL Engine • Powered by TensorTrade OMS. Deep Reinforcement Learning. Not financial advice.</i></div>", unsafe_allow_html=True)
