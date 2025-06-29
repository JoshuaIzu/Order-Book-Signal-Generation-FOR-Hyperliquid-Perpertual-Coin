import streamlit as st
import time
import pandas as pd
import numpy as np
import ccxt
import json
import os
from datetime import datetime
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from threading import Thread
from typing import Dict, List, Tuple, Union

# Set page config
st.set_page_config(
    page_title="Crypto Order Book Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: white;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stTextInput>div>div>input {
        color: #4CAF50;
    }
    .stSelectbox>div>div>select {
        color: #4CAF50;
    }
    .signal-bullish {
        background-color: #2e7d32;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .signal-bearish {
        background-color: #c62828;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .signal-neutral {
        background-color: #37474f;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .trade-log {
        font-family: monospace;
        background-color: #1E1E1E;
        padding: 10px;
        border-radius: 5px;
        max-height: 300px;
        overflow-y: auto;
    }
    </style>
    """, unsafe_allow_html=True)

class OrderBookStorage:
    def __init__(self, base_path: str = 'data/orderbook', max_history: int = 100):
        self.base_path = base_path
        self.max_history = max_history
        os.makedirs(base_path, exist_ok=True)
        self.recent_order_books = deque(maxlen=max_history)

    def store_order_book(self, order_book: Dict, symbol: str) -> str:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        clean_symbol = symbol.replace('/', '_').replace(':', '_')
        symbol_path = os.path.join(self.base_path, clean_symbol)
        os.makedirs(symbol_path, exist_ok=True)

        data_to_store = {
            'timestamp': timestamp,
            'symbol': symbol,
            'data': order_book
        }
        self.recent_order_books.append(data_to_store)

        filename = f"{clean_symbol}_{timestamp}.json"
        file_path = os.path.join(symbol_path, filename)

        with open(file_path, 'w') as f:
            json.dump(data_to_store, f, indent=4)

        return file_path

    def get_recent_order_books(self, n: int = None) -> List[Dict]:
        if n is None or n >= len(self.recent_order_books):
            return list(self.recent_order_books)
        return list(self.recent_order_books)[-n:]

class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1, dropout_rate: float = 0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        out = self.dropout(h_lstm[:, -1, :])  # Take the last hidden state
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

class CryptoOrderBook:
    def __init__(self, sequence_length: int = 10, feature_size: int = 40):
        self.pattern_detector = None
        self.sequence_length = sequence_length
        self.feature_size = feature_size
        self.last_predictions = deque(maxlen=5)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_lstm_model(self) -> LSTMModel:
        model = LSTMModel(self.feature_size, 64).to(self.device)
        return model

    def train_pattern_detector(self, data_sequences: List[np.ndarray], labels: np.ndarray, epochs=10, batch_size=32):
        X = torch.tensor(np.array(data_sequences), dtype=torch.float32).to(self.device)
        y = torch.tensor(np.array(labels), dtype=torch.float32).unsqueeze(1).to(self.device) # unsqueeze for binary cross entropy

        if self.pattern_detector is None:
            self.pattern_detector = self.build_lstm_model()

        criterion = nn.BCELoss() # Binary Cross Entropy Loss
        optimizer = optim.Adam(self.pattern_detector.parameters())

        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            for i, (inputs, targets) in enumerate(dataloader):
                optimizer.zero_grad()
                outputs = self.pattern_detector(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            st.toast(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")


    def prepare_sequence_data(self, order_books: List[Dict]) -> Tuple[List[np.ndarray], List[int]]:
        sequences = []
        labels = []

        if len(order_books) < self.sequence_length + 1:
            return sequences, labels

        for i in range(len(order_books) - self.sequence_length):
            sequence = order_books[i:i+self.sequence_length]
            next_book = order_books[i+self.sequence_length]

            sequence_features = [self.preprocess_order_book(ob) for ob in sequence]
            sequences.append(np.array(sequence_features))

            current_price = self.get_mid_price(sequence[-1])
            next_price = self.get_mid_price(next_book)
            labels.append(1 if next_price > current_price else 0)

        return sequences, labels

    def preprocess_order_book(self, order_book: Dict) -> np.ndarray:
        if 'data' in order_book:
            order_book = order_book['data']

        asks = np.array(order_book['asks'][:10])
        bids = np.array(order_book['bids'][:10])

        if asks.shape[0] < 10:
            padding = np.zeros((10 - asks.shape[0], 2))
            asks = np.vstack([asks, padding])
        if bids.shape[0] < 10:
            padding = np.zeros((10 - bids.shape[0], 2))
            bids = np.vstack([bids, padding])

        ask_prices = asks[:, 0]
        ask_volumes = asks[:, 1]
        bid_prices = bids[:, 0]
        bid_volumes = bids[:, 1]

        # Ensure features size consistency (20 prices + 20 volumes + 20 spreads = 60)
        # Note: The original code only calculated spread on the top 10,
        # but combined all 20 prices/volumes. Let's make sure `feature_size` is consistent.
        # If your intention was 40 features (20 price, 20 volume), then remove spreads or adjust.
        # Given the original Keras model `feature_size=40` but `feature_size=60` in OrderBookMonitor,
        # I'll stick to 60 for the PyTorch implementation to match.
        # If you meant 40, you might need to adjust what features you include.
        ask_spreads = np.diff(ask_prices, prepend=ask_prices[0])
        bid_spreads = np.diff(bid_prices, prepend=bid_prices[0])

        features = np.concatenate([
            ask_prices, ask_volumes,
            bid_prices, bid_volumes,
            ask_spreads, bid_spreads
        ])
        
        # Ensure the feature vector has the expected size
        if len(features) != self.feature_size:
            raise ValueError(f"Feature size mismatch. Expected {self.feature_size}, got {len(features)}")

        features = features / (np.mean(features) + 1e-8)
        return features

    def get_mid_price(self, order_book: Dict) -> float:
        if 'data' in order_book:
            order_book = order_book['data']
        best_ask = float(order_book['asks'][0][0])
        best_bid = float(order_book['bids'][0][0])
        return (best_ask + best_bid) / 2

    def detect_pattern(self, recent_order_books: List[Dict]) -> Tuple[str, float]:
        if self.pattern_detector is None:
            raise ValueError("Model has not been trained yet")

        if len(recent_order_books) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} order books for prediction")

        sequence = recent_order_books[-self.sequence_length:]
        sequence_features = [self.preprocess_order_book(ob) for ob in sequence]
        
        # Convert to PyTorch tensor and move to device
        X = torch.tensor(np.array([sequence_features]), dtype=torch.float32).to(self.device)

        self.pattern_detector.eval() # Set model to evaluation mode
        with torch.no_grad(): # Disable gradient calculation for inference
            prediction = self.pattern_detector(X).item() # Get scalar prediction
        self.pattern_detector.train() # Set model back to training mode

        self.last_predictions.append(prediction)

        smoothed_prediction = np.mean(self.last_predictions)
        confidence = abs(smoothed_prediction - 0.5) * 2

        if smoothed_prediction > 0.55:
            pattern = "Bullish"
        elif smoothed_prediction < 0.45:
            pattern = "Bearish"
        else:
            pattern = "Neutral"

        return pattern, confidence

class OrderBookMonitor:
    def __init__(self, symbol: str = 'VIRTUAL/USDC:USDC', check_interval: int = 5, training_interval: int = 3600):
        self.storage = OrderBookStorage(max_history=500)
        # Updated feature_size to 60 as per the concatenate in preprocess_order_book
        self.analyzer = CryptoOrderBook(sequence_length=10, feature_size=60) 
        self.exchange = ccxt.hyperliquid()
        self.check_interval = check_interval
        self.training_interval = training_interval
        self.last_pattern = None
        self.last_training_time = 0
        self.symbol = self._validate_symbol_format(symbol)
        self.running = False
        self.thread = None
        self.signals = deque(maxlen=6)
        self.trades = []
        
        # Initialize with some data
        self._collect_initial_data()

    def _validate_symbol_format(self, symbol: str) -> str:
        """Ensure symbol is in VIRTUAL/USDC:USDC format"""
        if ':' not in symbol:
            if '/' in symbol:
                base, quote = symbol.split('/')
                symbol = f"{base}/{quote}:{quote}"
            else:
                symbol = f"{symbol}/USDC:USDC"
        return symbol

    def _collect_initial_data(self):
        st.info("Collecting initial order book data (this may take a moment)...")
        for _ in range(30):
            try:
                order_book = self.exchange.fetch_order_book(self.symbol)
                self.storage.store_order_book(order_book, self.symbol)
                time.sleep(0.5) # Reduced sleep for faster initial collection
            except Exception as e:
                st.warning(f"Error collecting initial data: {e}. Retrying...")
                time.sleep(2)
        self._initial_training()

    def _initial_training(self):
        st.info("Performing initial model training...")
        order_books = self.storage.get_recent_order_books()
        if len(order_books) < 20: # Ensure enough data for initial training
            st.warning("Not enough data for initial training. Collecting more...")
            self._collect_initial_data() # Try collecting again if not enough
            return

        sequences, labels = self.analyzer.prepare_sequence_data(order_books)
        if sequences and labels:
            self.analyzer.train_pattern_detector(sequences, labels, epochs=5, batch_size=8)
            self.last_training_time = time.time()
            st.success("Initial model training complete.")
        else:
            st.warning("Could not prepare data for initial training. Check data collection.")


    def should_retrain(self) -> bool:
        current_time = time.time()
        return (current_time - self.last_training_time) > self.training_interval

    def retrain_model(self):
        st.info("Retraining model with new data...")
        order_books = self.storage.get_recent_order_books(100) # Use more recent data for retraining
        sequences, labels = self.analyzer.prepare_sequence_data(order_books)
        if sequences and labels:
            self.analyzer.train_pattern_detector(sequences, labels, epochs=3, batch_size=8)
            self.last_training_time = time.time()
            st.success("Model retraining complete.")
        else:
            st.warning("Not enough data to retrain model.")

    def get_current_price(self, order_book: dict) -> float:
        if 'data' in order_book:
            order_book = order_book['data']
        best_ask = float(order_book['asks'][0][0])
        best_bid = float(order_book['bids'][0][0])
        return (best_ask + best_bid) / 2

    def monitor_orderbook(self):
        self.running = True
        while self.running:
            try:
                order_book = self.exchange.fetch_order_book(self.symbol)
                self.storage.store_order_book(order_book, self.symbol)
                recent_books = self.storage.get_recent_order_books(15)

                if len(recent_books) >= self.analyzer.sequence_length:
                    current_pattern, confidence = self.analyzer.detect_pattern(recent_books)
                    current_price = self.get_current_price(order_book)
                    
                    # Create signal dictionary
                    signal = {
                        'timestamp': datetime.now().strftime('%H:%M:%S'),
                        'pattern': current_pattern,
                        'confidence': confidence,
                        'price': current_price,
                        'symbol': self.symbol
                    }
                    
                    # Add to signals
                    self.signals.append(signal)
                    
                    # Check for pattern change
                    if self.last_pattern and current_pattern != self.last_pattern:
                        st.toast(f"Pattern change detected: {self.last_pattern} -> {current_pattern}")
                    
                    self.last_pattern = current_pattern
                    
                    # Check for trading opportunity (confidence > 60%)
                    if confidence >= 0.6:
                        self.log_trade(signal)
                    
                    # Periodically retrain
                    if self.should_retrain():
                        self.retrain_model()

                time.sleep(self.check_interval)

            except Exception as e:
                st.error(f"Error in monitoring loop: {e}")
                time.sleep(self.check_interval * 2)

    def log_trade(self, signal):
        trade = {
            'timestamp': signal['timestamp'],
            'symbol': signal['symbol'],
            'action': 'BUY' if signal['pattern'] == 'Bullish' else 'SELL',
            'price': signal['price'],
            'confidence': signal['confidence'],
            'pattern': signal['pattern']
        }
        self.trades.append(trade)
        st.toast(f"Trade Signal: {trade['action']} {trade['symbol']} at {trade['price']:.4f} (Conf: {trade['confidence']*100:.1f}%)")

    def start(self):
        if not self.running:
            self.thread = Thread(target=self.monitor_orderbook)
            self.thread.daemon = True # Allow the thread to exit when main program exits
            self.thread.start()
            st.session_state.is_monitoring = True # Set a flag for UI

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=self.check_interval * 3) # Give it some time to stop
        st.session_state.is_monitoring = False # Clear the flag for UI

# Initialize session state
if 'monitor' not in st.session_state:
    st.session_state.monitor = None
if 'signals' not in st.session_state:
    st.session_state.signals = deque(maxlen=6)
if 'trades' not in st.session_state:
    st.session_state.trades = []
if 'is_monitoring' not in st.session_state:
    st.session_state.is_monitoring = False

# App UI
st.title("📊 Crypto Order Book Analyzer")
st.markdown("Analyze order book patterns and detect trading signals in real-time")

# Sidebar for controls
with st.sidebar:
    st.header("Configuration")
    symbol_input = st.text_input("Trading Pair (e.g., VIRTUAL/USDC:USDC)", value="VIRTUAL/USDC:USDC")
    check_interval = st.slider("Check Interval (seconds)", 1, 60, 5)
    
    if st.button("Start Monitoring", disabled=st.session_state.is_monitoring):
        if st.session_state.monitor:
            st.session_state.monitor.stop() # Ensure previous monitor is stopped cleanly
        st.session_state.monitor = OrderBookMonitor(
            symbol=symbol_input,
            check_interval=check_interval
        )
        st.session_state.monitor.start()
        st.success(f"Started monitoring {st.session_state.monitor.symbol}")
        st.experimental_rerun() # Rerun to update UI immediately

    if st.button("Stop Monitoring", disabled=not st.session_state.is_monitoring):
        if st.session_state.monitor:
            st.session_state.monitor.stop()
            st.session_state.monitor = None
            st.success("Monitoring stopped")
            st.experimental_rerun() # Rerun to update UI immediately
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This app analyzes cryptocurrency order books using LSTM neural networks 
    to detect patterns and generate trading signals.
    
    Symbol format: `TOKEN/QUOTE:SETTLEMENT` (e.g., `VIRTUAL/USDC:USDC`)
    """)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Real-time Signals")
    
    signals_placeholder = st.empty() # Placeholder for dynamic updates

with col2:
    st.header("Trade Log")
    
    trades_placeholder = st.empty() # Placeholder for dynamic updates

# Auto-refresh loop for signals and trades
if st.session_state.is_monitoring:
    # This loop will run as long as the app is running and monitoring is active
    # and will re-render the sections with the latest data.
    while st.session_state.monitor and st.session_state.monitor.running:
        with signals_placeholder.container():
            st.empty() # Clear previous content
            for signal in st.session_state.monitor.signals:
                if signal['pattern'] == "Bullish":
                    css_class = "signal-bullish"
                elif signal['pattern'] == "Bearish":
                    css_class = "signal-bearish"
                else:
                    css_class = "signal-neutral"
                
                st.markdown(f"""
                <div class="{css_class}">
                    <strong>{signal['timestamp']} - {signal['symbol']}</strong><br>
                    Pattern: {signal['pattern']}<br>
                    Confidence: {signal['confidence']*100:.1f}%<br>
                    Price: {signal['price']:.4f}
                </div>
                """, unsafe_allow_html=True)
        
        with trades_placeholder.container():
            st.empty() # Clear previous content
            if st.session_state.monitor.trades:
                st.markdown("""
                <div class="trade-log">
                """, unsafe_allow_html=True)
                
                for trade in reversed(st.session_state.monitor.trades[-10:]):  # Show last 10 trades
                    color = "green" if trade['action'] == 'BUY' else "red"
                    st.markdown(f"""
                    <span style="color: {color};">
                    {trade['timestamp']} - {trade['action']} {trade['symbol']} @ {trade['price']:.4f}<br>
                    Confidence: {trade['confidence']*100:.1f}% ({trade['pattern']})
                    </span><br>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("No trades yet")
        
        time.sleep(2) # Refresh every 2 seconds

else:
    with col1:
        st.info("Start monitoring to see signals")
    with col2:
        st.info("Start monitoring to see trades")

# Clean up when app is closed (Streamlit handles this somewhat automatically, but good practice)
def cleanup_on_exit():
    if st.session_state.monitor:
        st.session_state.monitor.stop()
        print("Monitoring stopped on app exit.") # For debugging
import atexit
atexit.register(cleanup_on_exit)
