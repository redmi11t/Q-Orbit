"""
Simplified QAOA Optimizer for Portfolio Optimization
Works with Qiskit Aer backend directly (no primitives)
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from scipy.optimize import minimize

# Qiskit imports
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

try:
    from .qubo_formulation import PortfolioQUBO
except ImportError:
    from qubo_formulation import PortfolioQUBO


class QAOAOptimizer:
    """
    Simplified QAOA for Portfolio Selection
    Uses direct Qiskit Aer backend execution
    """
    
    def __init__(
        self,
        num_layers: int = 2,
        max_iterations: int = 50,
        backend_name: str = 'qasm_simulator'
    ):
        """Initialize QAOA optimizer"""
        self.num_layers = num_layers
        self.max_iterations = max_iterations
        self.backend = AerSimulator()
        
        self.optimal_params = None
        self.optimal_bitstring = None
        self.optimization_history = []
        self.iteration_count = 0
    
    def optimize(
        self,
        returns: pd.DataFrame,
        budget: int = 5,
        risk_factor: float = 1.0,
        return_factor: float = 0.5,
        budget_penalty: float = 10.0,
        precomputed_Q: np.ndarray = None
    ) -> Tuple[List[str], np.ndarray, Dict]:
        """
        Run QAOA optimization

        Args:
            returns: Historical returns DataFrame
            budget: Number of stocks to select
            risk_factor: QUBO risk weight
            return_factor: QUBO return weight
            budget_penalty: QUBO budget constraint penalty
            precomputed_Q: Optional pre-built (e.g. sentiment-adjusted) QUBO matrix.
                           When supplied the internal QUBO construction is skipped.

        Returns:
            Tuple of (selected_tickers, weights, info_dict)
        """
        print(f"\n{'='*70}")
        print(f"QAOA PORTFOLIO OPTIMIZATION")
        print(f"{'='*70}")
        print(f"Portfolio size: {len(returns.columns)} stocks")
        print(f"Budget: Select {budget} stocks")
        print(f"QAOA layers: {self.num_layers}")

        # Formulate QUBO — use caller-supplied matrix if provided
        if precomputed_Q is not None:
            print(f"\n[1/3] Using precomputed (sentiment-adjusted) QUBO matrix...")
            qubo = PortfolioQUBO(risk_factor, return_factor, budget_penalty)
            qubo.tickers = returns.columns.tolist()
            qubo.num_assets = len(returns.columns)
            qubo.Q = precomputed_Q
            Q = precomputed_Q
        else:
            print(f"\n[1/3] Formulating QUBO problem...")
            qubo = PortfolioQUBO(risk_factor, return_factor, budget_penalty)
            Q = qubo.formulate(returns, budget)

        num_qubits = len(returns.columns)
        
        # Run QAOA
        print("\n[2/3] Running QAOA on quantum simulator...")
        initial_params = np.random.uniform(0, 2*np.pi, 2 * self.num_layers)
        optimal_params, optimal_value = self._run_qaoa(Q, num_qubits, initial_params)
        bitstring = self.optimal_bitstring
        
        # Decode solution
        print("\n[3/3] Decoding solution and refining weights...")
        selected_indices, selected_tickers, _ = qubo.decode_solution(bitstring)
        
        # If no stocks selected (rare), default to all
        if not selected_tickers:
            selected_tickers = returns.columns.tolist()
            selected_indices = list(range(len(returns.columns)))
            
        # Post-selection weight refinement: Run classical Markowitz on selected subset
        # This is a 'Best of Both Worlds' approach: Quantum for selection, Classical for sizing.
        from classical.baseline import MarkowitzOptimizer
        subset_returns = returns[selected_tickers]
        subset_opt = MarkowitzOptimizer()
        
        try:
            # Try to get optimal weights for the subset
            refined_weights_subset = subset_opt.optimize_min_variance(subset_returns)
            
            # Map subset weights back to full asset list
            weights = np.zeros(len(returns.columns))
            for i, ticker in enumerate(returns.columns):
                if ticker in refined_weights_subset.index:
                    weights[i] = refined_weights_subset[ticker]
        except Exception as e:
            print(f"Warning: Classical refinement failed ({e}), falling back to equal weights.")
            weights = np.zeros(len(returns.columns))
            for i in selected_indices:
                weights[i] = 1.0 / len(selected_indices)
        
        # Evaluate objective for the QAOA selected bitstring
        objective = qubo.evaluate_objective(bitstring)
        
        print(f"\n{'='*70}")
        print(f"QAOA COMPLETE")
        print(f"{'='*70}")
        print(f"Iterations: {self.iteration_count}")
        print(f"Selected: {', '.join(selected_tickers)}")
        
        info = {
            'iterations': self.iteration_count,
            'objective': objective,
            'num_layers': self.num_layers,
            'optimal_params': optimal_params,
            'bitstring': bitstring,
            'weights': weights  # Include refined weights in info
        }
        
        return selected_tickers, weights, info
    
    def _build_qaoa_circuit(
        self,
        Q: np.ndarray,
        num_qubits: int,
        params: np.ndarray
    ) -> QuantumCircuit:
        """Build QAOA circuit"""
        qc = QuantumCircuit(num_qubits, num_qubits)
        
        # Initial equal superposition
        qc.h(range(num_qubits))
        
        gammas = params[:self.num_layers]
        betas = params[self.num_layers:]
        
        # QAOA layers
        for p in range(self.num_layers):
            # Cost Hamiltonian (from QUBO)
            # Diagonal terms
            for i in range(num_qubits):
                if abs(Q[i, i]) > 1e-10:
                    qc.rz(2 * gammas[p] * Q[i, i], i)
            
            # Off-diagonal terms (two-qubit gates)
            for i in range(num_qubits):
                for j in range(i + 1, num_qubits):
                    if abs(Q[i, j]) > 1e-10:
                        # ZZ interaction
                        qc.cx(i, j)
                        qc.rz(2 * gammas[p] * Q[i, j], j)
                        qc.cx(i, j)
            
            # Mixer Hamiltonian
            for qubit in range(num_qubits):
                qc.rx(2 * betas[p], qubit)
        
        # Measurement
        qc.measure(range(num_qubits), range(num_qubits))
        
        return qc
    
    def _run_qaoa(
        self,
        Q: np.ndarray,
        num_qubits: int,
        initial_params: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Execute QAOA with classical optimization"""
        self.optimization_history = []
        self.iteration_count = 0
        
        def cost_function(params):
            """Evaluate QAOA circuit"""
            self.iteration_count += 1
            
            # Build and run circuit
            qc = self._build_qaoa_circuit(Q, num_qubits, params)
            transpiled_qc = transpile(qc, self.backend)
            job = self.backend.run(transpiled_qc, shots=1024)
            result = job.result()
            counts = result.get_counts()
            
            # Calculate expectation value
            expectation = 0.0
            best_bitstring = None
            best_count = 0
            
            for bitstring, count in counts.items():
                # Reverse bitstring (Qiskit convention)
                bitstring_reversed = bitstring[::-1]
                
                # Track most frequent bitstring
                if count > best_count:
                    best_count = count
                    best_bitstring = bitstring_reversed
                
                # Evaluate QUBO objective
                x = np.array([int(b) for b in bitstring_reversed])
                energy = x @ Q @ x
                
                probability = count / 1024
                expectation += probability * energy
            
            if best_bitstring:
                self.optimal_bitstring = best_bitstring
            
            self.optimization_history.append(expectation)
            
            if self.iteration_count % 10 == 0:
                print(f"  Iteration {self.iteration_count}: expectation = {expectation:.4f}")
            
            return expectation
        
        # Classical optimization
        print(f"  Starting classical optimization (COBYLA)...")
        result = minimize(
            cost_function,
            initial_params,
            method='COBYLA',
            options={'maxiter': self.max_iterations, 'disp': False}
        )
        
        self.optimal_params = result.x
        return result.x, result.fun
    
    def get_circuit_depth(self, num_qubits: int) -> int:
        """Calculate circuit depth"""
        # Rough estimate
        gates_per_layer = num_qubits + num_qubits * (num_qubits - 1)
        return self.num_layers * gates_per_layer


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import sys
    import os
    
    # Adding src path to find classical module
    src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    print("Testing QAOA Optimizer...")
    
    # Test data
    np.random.seed(42)
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA']
    n_stocks = len(tickers)
    n_days = 252
    
    mean_returns = np.array([0.25, 0.22, 0.20, 0.35, 0.15]) / 252
    volatilities = np.array([0.30, 0.25, 0.28, 0.45, 0.60]) / np.sqrt(252)
    corr = np.eye(n_stocks) + 0.3 * (np.ones((n_stocks, n_stocks)) - np.eye(n_stocks))
    
    cov = np.outer(volatilities, volatilities) * corr
    returns_array = np.random.multivariate_normal(mean_returns, cov, n_days)
    returns_df = pd.DataFrame(returns_array, columns=tickers)
    
    # Run QAOA
    qaoa = QAOAOptimizer(num_layers=2, max_iterations=30)
    selected, weights, info = qaoa.optimize(returns_df, budget=3)
    
    print(f"\nSelected: {selected}")
    print(f"Weights: {weights}")
    print("\nTest complete!")
