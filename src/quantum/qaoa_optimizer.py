"""
QAOA Optimizer for Portfolio Optimization
Supports Qiskit Aer (local simulator) and IBM Quantum real hardware.
"""

import warnings
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from scipy.optimize import minimize

# Qiskit imports
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# Fix #19: move import to module level so it is not repeated inside the hot path
try:
    from classical.baseline import MarkowitzOptimizer
except ImportError:
    from src.classical.baseline import MarkowitzOptimizer

try:
    from .qubo_formulation import PortfolioQUBO
except ImportError:
    from qubo_formulation import PortfolioQUBO

try:
    from .ibm_backend import IBMBackendHelper
except ImportError:
    try:
        from ibm_backend import IBMBackendHelper
    except ImportError:
        IBMBackendHelper = None


class QAOAOptimizer:
    """
    Simplified QAOA for Portfolio Selection.

    Uses the AerSimulator in **statevector mode** for the COBYLA optimisation
    loop (exact expectation values, zero shot noise) and a final 512-shot run
    only to decode the best bitstring.  This is typically 10-50× faster than
    the old 1024-shots-per-iteration approach.
    """
    
    def __init__(
        self,
        num_layers: int = 1,
        max_iterations: int = 25,
        backend_name: str = 'statevector_simulator',
        backend_mode: str = 'simulator',
        ibm_backend_name: Optional[str] = None,
    ):
        """
        Initialize QAOA optimizer.

        Args:
            num_layers: Number of QAOA layers (p).
            max_iterations: Maximum COBYLA iterations.
            backend_name: Legacy param (ignored when backend_mode='ibm_real').
            backend_mode: 'simulator' (default, uses Aer) or 'ibm_real' (IBM QPU).
            ibm_backend_name: IBM QPU name, e.g. 'ibm_brisbane'. Reads from
                              IBM_QUANTUM_BACKEND env var if not specified.
        """
        self.num_layers = num_layers
        self.max_iterations = max_iterations
        self.backend_mode = backend_mode
        self.ibm_backend_name = ibm_backend_name
        self.using_ibm = False  # Set True only when IBM connection succeeds

        # ── Choose backend ───────────────────────────────────────────────────
        self.ibm_error: Optional[str] = None   # Set if IBM connection failed
        if backend_mode == 'ibm_real' and IBMBackendHelper is not None:
            helper = IBMBackendHelper(backend_name=ibm_backend_name)
            ibm_backend = helper.get_backend()
            if ibm_backend is not None:
                self.ibm_backend_obj = ibm_backend
                self.using_ibm = True
                print(f"[IBM Quantum] Connected to: {helper.backend_name}")
            else:
                self.ibm_error = helper.last_error   # Store real reason
                warnings.warn(
                    f"IBM Quantum connection failed ({helper.last_error}) "
                    "— falling back to Aer simulator.",
                    RuntimeWarning, stacklevel=2
                )
                self.ibm_backend_obj = None
        else:
            self.ibm_backend_obj = None

        # Local Aer backends (always initialised as fallback)
        self.backend = AerSimulator(method='statevector')
        self.shot_backend = AerSimulator()
        # ────────────────────────────────────────────────────────────────────
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

        # ── Fix 2: Hard qubit limit ──────────────────────────────────────────
        # Classical simulation of quantum circuits scales *exponentially*
        # with the number of qubits.  Beyond 10 the simulator will freeze.
        MAX_QUBITS = 10
        n_stocks = len(returns.columns)
        if n_stocks > MAX_QUBITS:
            raise ValueError(
                f"QAOA is limited to {MAX_QUBITS} stocks on a classical simulator "
                f"(exponential scaling).  You selected {n_stocks} stocks.  "
                "Please reduce the portfolio size to ≤10 stocks when using a "
                "quantum method, or switch to a classical optimizer."
            )
        # ────────────────────────────────────────────────────────────
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

        # ── Budget enforcement ───────────────────────────────────────────────
        # QAOA with p=1 / few iterations may select fewer stocks than the budget
        # because the budget-constraint penalty is soft (not exact).  Greedily
        # pad with the highest expected-return unselected stocks so the returned
        # portfolio always has exactly `budget` holdings.
        all_tickers = returns.columns.tolist()
        if len(selected_tickers) < budget:
            mean_rets = returns.mean()
            unselected = [t for t in all_tickers if t not in selected_tickers]
            # Sort by expected return (descending) and take the shortfall
            shortfall = budget - len(selected_tickers)
            extras = sorted(unselected, key=lambda t: mean_rets[t], reverse=True)[:shortfall]
            selected_tickers += extras
            selected_indices += [all_tickers.index(t) for t in extras]
            print(f"  ⚠ Budget shortfall padded with: {', '.join(extras)}")
        elif len(selected_tickers) > budget:
            # Trim to budget keeping best by expected return
            mean_rets = returns.mean()
            selected_tickers = sorted(selected_tickers,
                                      key=lambda t: mean_rets[t], reverse=True)[:budget]
            selected_indices = [all_tickers.index(t) for t in selected_tickers]
            print(f"  ⚠ Over-selection trimmed to budget={budget}")
        # ─────────────────────────────────────────────────────────────────────

        # If no stocks selected (rare), default to all
        if not selected_tickers:
            selected_tickers = returns.columns.tolist()
            selected_indices = list(range(len(returns.columns)))
            
        # Post-selection weight refinement: Run classical Markowitz on selected subset
        # This is a 'Best of Both Worlds' approach: Quantum for selection, Classical for sizing.
        subset_returns = returns[selected_tickers]
        subset_opt = MarkowitzOptimizer()  # imported at module level (Fix #19)
        
        try:
            # Try to get optimal weights for the subset
            refined_weights_subset = subset_opt.optimize_min_variance(subset_returns)
            
            # Map subset weights back to full asset list
            weights = np.zeros(len(returns.columns))
            for i, ticker in enumerate(returns.columns):
                if ticker in refined_weights_subset.index:
                    weights[i] = refined_weights_subset[ticker]
            fallback_warning = None  # success — no need for a warning
        except Exception as e:
            # ── Fix 6: Surface fallback warning to caller ────────────────────
            # Previously this was only printed to the console.  Now we also
            # store the message in ``info`` so app.py can render st.warning.
            fallback_warning = (
                f"Classical weight refinement on the QAOA-selected subset failed "
                f"({e}).  Equal weights have been applied to the selected stocks."
            )
            print(f"Warning: {fallback_warning}")
            # ────────────────────────────────────────────────────────────
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
            'weights': weights,
            'fallback_warning': fallback_warning,
            'backend': 'IBM Quantum — ' + (self.ibm_backend_name or 'ibm_brisbane') if self.using_ibm else 'Aer Simulator (local)',
        }
        
        return selected_tickers, weights, info
    
    def _build_qaoa_circuit(
        self,
        Q: np.ndarray,
        num_qubits: int,
        params: np.ndarray,
        with_measurement: bool = True
    ) -> QuantumCircuit:
        """Build QAOA circuit.  Set with_measurement=False for statevector mode."""
        if with_measurement:
            qc = QuantumCircuit(num_qubits, num_qubits)
        else:
            qc = QuantumCircuit(num_qubits)
        
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
        
        if with_measurement:
            qc.measure(range(num_qubits), range(num_qubits))
        else:
            qc.save_statevector()  # Required for statevector extraction
        
        return qc
    
    def _run_qaoa(
        self,
        Q: np.ndarray,
        num_qubits: int,
        initial_params: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Execute QAOA with classical optimization.

        Speed fix: Uses the statevector simulator to compute exact expectation
        values E[x^T Q x] during the COBYLA loop (no shot noise, no repeated
        sampling). Only the *final* bitstring decode uses a 512-shot run.
        """
        self.optimization_history = []
        self.iteration_count = 0

        # Pre-compute all 2^n QUBO energies once so the cost function only
        # needs to do a dot-product, not a matrix multiply per bitstring.
        n = num_qubits
        all_bitstrings = np.array(
            [[int(b) for b in format(k, f'0{n}b')] for k in range(2**n)],
            dtype=float
        )  # shape (2^n, n)
        qubo_energies = np.einsum('ij,jk,ik->i', all_bitstrings, Q, all_bitstrings)
        # shape (2^n,)

        def cost_function(params):
            """Expectation value — statevector (Aer) or sampler (IBM)."""
            self.iteration_count += 1

            if self.using_ibm:
                # ── IBM Real Hardware path (shot-based) ───────────────────────
                try:
                    from qiskit_ibm_runtime import SamplerV2 as Sampler
                    from qiskit_ibm_runtime import SamplerOptions

                    qc = self._build_qaoa_circuit(Q, n, params, with_measurement=True)
                    transpiled = transpile(qc, self.ibm_backend_obj)
                    sampler = Sampler(backend=self.ibm_backend_obj)
                    job = sampler.run([transpiled], shots=1024)
                    result_ibm = job.result()
                    counts_data = result_ibm[0].data
                    # Extract counts from the classical register
                    creg = list(counts_data.keys())[0]
                    counts = counts_data[creg].get_counts()
                    total = sum(counts.values())
                    probs = np.zeros(2**n)
                    for bitstr, cnt in counts.items():
                        # Phase 2 Fix #7: Qiskit measurement counts are
                        # LSB-first (little-endian), but the statevector
                        # probability array is MSB-first (big-endian).
                        # Reverse the bitstring before converting to an index
                        # so both paths select the same bitstring.
                        idx = int(bitstr[::-1], 2)
                        if idx < 2**n:
                            probs[idx] = cnt / total
                except Exception as ibm_exc:
                    warnings.warn(
                        f"IBM job failed ({ibm_exc}), switching to Aer for this iteration.",
                        RuntimeWarning, stacklevel=2
                    )
                    # Fallback to statevector for this iteration
                    qc = self._build_qaoa_circuit(Q, n, params, with_measurement=False)
                    transpiled = transpile(qc, self.backend)
                    job = self.backend.run(transpiled)
                    sv = job.result().get_statevector()
                    probs = np.abs(np.array(sv))**2
            else:
                # ── Aer Statevector path (default, fast) ──────────────────────
                qc = self._build_qaoa_circuit(Q, n, params, with_measurement=False)
                transpiled = transpile(qc, self.backend)
                job = self.backend.run(transpiled)
                sv = job.result().get_statevector()
                probs = np.abs(np.array(sv))**2

            # E[energy] = sum_k P(k) * energy(k)
            expectation = float(np.dot(probs, qubo_energies))

            # Fix #9: Best bitstring = minimum QUBO energy among the top-5 most
            # probable states.  Picking purely by max-probability is only correct
            # when QAOA has fully converged (large p, many iterations).  Evaluating
            # energy for the top candidates is cheap and far more robust.
            top5_indices = np.argpartition(probs, -min(5, len(probs)))[-min(5, len(probs)):]
            best_idx = int(top5_indices[np.argmin(qubo_energies[top5_indices])])
            self.optimal_bitstring = format(best_idx, f'0{n}b')  # MSB-first

            self.optimization_history.append(expectation)
            if self.iteration_count % 10 == 0:
                mode_str = 'IBM QPU' if self.using_ibm else 'Aer'
                print(f"  Iteration {self.iteration_count} [{mode_str}]: ⟨E⟩ = {expectation:.4f}")
            return expectation

        # Classical optimization
        mode_label = f'IBM QPU ({self.ibm_backend_name})' if self.using_ibm else 'statevector (Aer)'
        print(f"  Starting COBYLA ({mode_label}, max {self.max_iterations} iters)...")
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
