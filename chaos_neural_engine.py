import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Optional

# --- Configuration ---
SEED = 42
np.random.seed(SEED)
plt.style.use('dark_background')  # Cyberpunk scientific aesthetic

class LorenzSystem:
    """
    Simulates the Lorenz 63 system (The Butterfly Effect).
    Equations:
    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z
    """
    def __init__(self, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0/3.0):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def derivatives(self, state: Tuple[float, float, float], t: float) -> Tuple[float, float, float]:
        x, y, z = state
        dx_dt = self.sigma * (y - x)
        dy_dt = x * (self.rho - z) - y
        dz_dt = x * y - self.beta * z
        return dx_dt, dy_dt, dz_dt

    def generate_data(self, initial_state: Tuple[float, float, float], 
                      steps: int, dt: float) -> np.ndarray:
        """Generates time-series data using ODE integration."""
        t = np.arange(0, steps * dt, dt)
        return odeint(self.derivatives, initial_state, t)


class EchoStateNetwork:
    """
    A Reservoir Computer implementation.
    
    Concept: A fixed, randomized Recurrent Neural Network (Reservoir) projects 
    input into a high-dimensional space. We only train the 'Readout' layer 
    using Ridge Regression.
    """
    def __init__(self, input_dim: int, reservoir_size: int, spectral_radius: float = 0.95, 
                 sparsity: float = 0.1, leak_rate: float = 0.2, regularization: float = 1e-7):
        
        self.input_dim = input_dim
        self.reservoir_size = reservoir_size
        self.leak_rate = leak_rate
        self.regularization = regularization

        # 1. Input Weights (Win): Maps input to reservoir (Fixed, Random)
        self.Win = np.random.uniform(-0.5, 0.5, (reservoir_size, input_dim))

        # 2. Reservoir Weights (W): Internal connections (Fixed, Random, Sparse)
        # We create a sparse matrix to simulate sparse neural connectivity
        self.W = np.random.uniform(-1, 1, (reservoir_size, reservoir_size))
        mask = np.random.rand(reservoir_size, reservoir_size) > sparsity
        self.W[mask] = 0.0
        
        # Spectral Radius scaling (Crucial for ESN stability)
        # We ensure the echo "fades" appropriately to avoid exploding gradients
        eigenvalues = np.linalg.eigvals(self.W)
        max_eigenvalue = np.max(np.abs(eigenvalues))
        if max_eigenvalue > 0:
            self.W = self.W * (spectral_radius / max_eigenvalue)

        # 3. Output Weights (Wout): Learned via Regression
        self.Wout = None 
        
        # Internal state vector
        self.x = np.zeros((reservoir_size, 1))

    def _update_state(self, u: np.ndarray):
        """Updates the internal reservoir state x(t)."""
        # Non-linear activation (tanh) is the core of the "computing"
        pre_activation = np.dot(self.Win, u) + np.dot(self.W, self.x)
        self.x = (1 - self.leak_rate) * self.x + self.leak_rate * np.tanh(pre_activation)

    def train(self, inputs: np.ndarray, targets: np.ndarray, washout: int = 100):
        """
        Trains the readout layer using Ridge Regression.
        
        Args:
            washout: Number of initial timesteps to discard (warm-up phase)
                     to ensure the reservoir state is synchronized with input.
        """
        T = inputs.shape[0]
        # Collect all reservoir states during the training sequence
        X_states = np.zeros((self.reservoir_size, T))

        print(f"[*] Training Reservoir on {T} timesteps (Washout: {washout})...")
        
        for t in range(T):
            u = inputs[t].reshape(-1, 1)
            self._update_state(u)
            X_states[:, t] = self.x.flatten()

        # Discard the washout period (initial transient states)
        X_final = X_states[:, washout:]
        Y_final = targets[washout:]

        # Ridge Regression: Wout = Y * X^T * (X * X^T + beta * I)^-1
        X_T = X_final.T
        Y_T = Y_final
        
        # Solving linear system for speed and stability
        # (X * X^T + reg * I) * Wout^T = X * Y^T  <-- Math formulation
        # We transpose everything to match numpy's shape conventions
        
        # X_final is (Reservoir, Time), Y_final is (Time, Output)
        # Standard Ridge equation: W_out = Y X^T (X X^T + lambda I)^-1
        
        # Calculate X * X^T
        XXT = np.dot(X_final, X_final.T)
        
        # Add regularization to diagonal
        ridge = self.regularization * np.eye(self.reservoir_size)
        
        # Calculate Y * X^T
        YXT = np.dot(Y_final.T, X_final.T)
        
        # Solve (XXT + ridge) * W_out.T = (YXT).T -> W_out = YXT * inv(XXT + ridge)
        self.Wout = np.linalg.solve(XXT + ridge, YXT.T).T
        
        print("[*] Training Complete.")

    def predict(self, initial_input: np.ndarray, steps: int) -> np.ndarray:
        """
        Generative Mode: The network feeds its own output back into itself 
        to 'dream' the future trajectory.
        """
        predictions = np.zeros((steps, self.input_dim))
        current_input = initial_input.reshape(-1, 1)
        
        print(f"[*] Generating {steps} future timesteps...")
        
        for t in range(steps):
            self._update_state(current_input)
            # Wout projects high-dim reservoir state back to 3D space
            prediction = np.dot(self.Wout, self.x)
            predictions[t] = prediction.flatten()
            current_input = prediction # Feedback loop
            
        return predictions


def visualize_results(truth: np.ndarray, prediction: np.ndarray, train_len: int):
    """Generates a high-quality 3D plot comparing Truth vs AI Hallucination."""
    fig = plt.figure(figsize=(16, 8))
    
    # Plot 1: The Ground Truth (Physics)
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(truth[:, 0], truth[:, 1], truth[:, 2], lw=0.6, color='#00ffcc', alpha=0.8)
    ax1.set_title("Ground Truth (Lorenz Equations)", fontsize=14, color='white')
    ax1.set_axis_off() # Cleaner look

    # Plot 2: The Neural Prediction
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Plot the part the network saw during training (ghosted)
    ax2.plot(truth[:train_len, 0], truth[:train_len, 1], truth[:train_len, 2], 
             lw=0.5, color='gray', alpha=0.3, label='Training Phase')
    
    # Plot the part the network hallucinated (prediction)
    # We shift index to align visual flow
    ax2.plot(prediction[:, 0], prediction[:, 1], prediction[:, 2], 
             lw=0.8, color='#ff0055', alpha=0.9, label='Neural Generation')
    
    ax2.set_title("Reservoir Network Prediction", fontsize=14, color='white')
    ax2.legend()
    ax2.set_axis_off()

    plt.tight_layout()
    print("[*] Displaying Plot. Close window to exit.")
    plt.show()

def main():
    # 1. Data Generation
    dt = 0.02  # Slightly larger step for clearer dynamics
    total_steps = 8000
    train_steps = 5000 
    test_steps = 3000  
    washout = 200 # Ignore first 200 steps to let reservoir synchronize
    
    lorenz = LorenzSystem()
    data = lorenz.generate_data((1.0, 1.0, 1.0), total_steps, dt)

    # Normalize data (Crucial for Neural Networks)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    data_norm = (data - mean) / std

    # Prepare inputs and targets
    # Input: State at t
    # Target: State at t+1 (Next step prediction)
    X = data_norm[:train_steps]
    Y = data_norm[1:train_steps+1]

    # 2. Model Initialization
    # Tuned parameters for stability
    esn = EchoStateNetwork(
        input_dim=3, 
        reservoir_size=1000,   # Larger reservoir for better accuracy
        spectral_radius=0.95,  # < 1.0 ensures "Echo State Property" (Stability)
        leak_rate=0.15,        # Slower leak rate for smoother dynamics
        regularization=1e-7
    )

    # 3. Train
    esn.train(X, Y, washout=washout)

    # 4. Generative Prediction (Closed Loop)
    # Start predicting from the end of training data
    last_train_point = data_norm[train_steps]
    prediction_norm = esn.predict(last_train_point, test_steps)

    # Denormalize for plotting
    prediction = (prediction_norm * std) + mean
    truth = data # Original scale

    # 5. Visualize
    visualize_results(truth, prediction, train_steps)

if __name__ == "__main__":
    main()