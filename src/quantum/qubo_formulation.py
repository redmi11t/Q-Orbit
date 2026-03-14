"""
QUBO Formulation for Portfolio Optimization
Converts portfolio optimization problem to Quadratic Unconstrained Binary Optimization format
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


class PortfolioQUBO:
    """
    Convert portfolio optimization to QUBO formulation for quantum algorithms
    
    QUBO Objective: minimize x^T Q x where x are binary variables (0 or 1)
    
    Portfolio Objective:
        minimize: risk_factor * Σᵢⱼ xᵢ Covᵢⱼ xⱼ - return_factor * Σᵢ xᵢ Rᵢ + penalty * (Σᵢ xᵢ - budget)²
    
    Where:
        xᵢ = 1 if stock i is selected, 0 otherwise
        Covᵢⱼ = covariance matrix (risk)
        Rᵢ = expected returns
        budget = desired number of stocks to select
    """
    
    def __init__(
        self,
        risk_factor: float = 1.0,
        return_factor: float = 0.5,
        budget_penalty: float = 10.0
    ):
        """
        Initialize QUBO formulator
        
        Args:
            risk_factor: Weight for risk term (higher = more conservative)
            return_factor: Weight for return term (higher = more growth-oriented)
            budget_penalty: Penalty for violating budget constraint (should be >> other factors)
        """
        self.risk_factor = risk_factor
        self.return_factor = return_factor
        self.budget_penalty = budget_penalty
        
        self.Q = None  # QUBO matrix
        self.tickers = None
        self.num_assets = 0
    
    def formulate(
        self,
        returns: pd.DataFrame,
        budget: int = 5
    ) -> np.ndarray:
        """
        Convert portfolio optimization to QUBO matrix
        
        Args:
            returns: DataFrame of historical returns
            budget: Number of stocks to select
            
        Returns:
            Q matrix where objective = x^T Q x
        """
        self.tickers = returns.columns.tolist()
        self.num_assets = len(self.tickers)
        
        # Calculate portfolio statistics
        mean_returns = returns.mean().values * 252  # Annualized
        cov_matrix = returns.cov().values * 252  # Annualized
        
        # Initialize QUBO matrix
        Q = np.zeros((self.num_assets, self.num_assets))
        
        # 1. Risk term: risk_factor * Σᵢⱼ xᵢ Covᵢⱼ xⱼ
        Q += self.risk_factor * cov_matrix
        
        # 2. Return term: -return_factor * Σᵢ xᵢ Rᵢ
        # This goes on diagonal since Σᵢ xᵢ Rᵢ = Σᵢ Rᵢ xᵢ²  (and xᵢ² = xᵢ for binary)
        np.fill_diagonal(Q, Q.diagonal() - self.return_factor * mean_returns)
        
        # 3. Budget constraint: penalty * (Σᵢ xᵢ - budget)²
        # Expand: penalty * (Σᵢ xᵢ² - 2*budget*Σᵢ xᵢ + budget²)
        #       = penalty * (Σᵢ xᵢ - 2*budget*Σᵢ xᵢ + budget²)  [since xᵢ² = xᵢ]
        #       = penalty * ((1 - 2*budget)*Σᵢ xᵢ + budget²)
        
        # Diagonal term: penalty * (1 - 2*budget) for each xᵢ
        np.fill_diagonal(Q, Q.diagonal() + self.budget_penalty * (1 - 2 * budget))
        
        # Off-diagonal term: 2 * penalty for each xᵢ*xⱼ (from expanding (Σxᵢ)²)
        # Add penalty to all off-diagonal elements
        Q += self.budget_penalty * (1 - np.eye(self.num_assets))
        
        self.Q = Q
        return Q
    
    def decode_solution(self, bitstring: str) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """
        Convert binary solution to portfolio
        
        Args:
            bitstring: Binary string (e.g., "10110" means assets 0, 2, 3 selected)
            
        Returns:
            Tuple of (selected_indices, selected_tickers, weights)
        """
        if len(bitstring) != self.num_assets:
            raise ValueError(f"Bitstring length {len(bitstring)} doesn't match num_assets {self.num_assets}")
        
        # Convert bitstring to binary array
        selection = np.array([int(b) for b in bitstring])
        
        # Get selected stocks
        selected_indices = np.where(selection == 1)[0]
        selected_tickers = [self.tickers[i] for i in selected_indices]
        
        # Calculate weights (equal weight for selected stocks)
        weights = np.zeros(self.num_assets)
        if len(selected_indices) > 0:
            weights[selected_indices] = 1.0 / len(selected_indices)
        
        return selected_indices, selected_tickers, weights
    
    def evaluate_objective(self, bitstring: str) -> float:
        """
        Evaluate QUBO objective for a given solution
        
        Args:
            bitstring: Binary solution
            
        Returns:
            Objective value (lower is better)
        """
        x = np.array([int(b) for b in bitstring])
        return x @ self.Q @ x


def qubo_to_ising(Q: np.ndarray) -> Tuple[Dict, float]:
    """
    Convert QUBO to Ising Hamiltonian for QAOA
    
    QUBO uses binary variables xᵢ ∈ {0, 1}
    Ising uses spin variables sᵢ ∈ {-1, +1}
    
    Transformation: xᵢ = (1 - sᵢ) / 2
    
    Args:
        Q: QUBO matrix
        
    Returns:
        Tuple of (ising_dict, offset) where:
            ising_dict: {(i,j): coefficient} for Ising Hamiltonian
            offset: Constant offset term
    """
    n = Q.shape[0]
    ising_dict = {}
    offset = 0.0
    
    # Convert QUBO to Ising
    # x^T Q x with xᵢ = (1 - sᵢ)/2
    # = (1/4) * Σᵢⱼ Qᵢⱼ (1 - sᵢ)(1 - sⱼ)
    # = (1/4) * Σᵢⱼ Qᵢⱼ (1 - sᵢ - sⱼ + sᵢsⱼ)
    
    for i in range(n):
        for j in range(i, n):
            q_val = Q[i, j]
            if i != j:
                q_val += Q[j, i]  # Symmetrize
            
            if abs(q_val) > 1e-10:  # Skip near-zero terms
                if i == j:
                    # Diagonal term: (1/4) * Qᵢᵢ(1 - 2sᵢ + sᵢ²) = (1/4) * Qᵢᵢ(2 - 2sᵢ)
                    # = (1/2) * Qᵢᵢ - (1/2) * Qᵢᵢ sᵢ
                    offset += 0.5 * q_val
                    ising_dict[(i,)] = ising_dict.get((i,), 0.0) - 0.5 * q_val
                else:
                    # Off-diagonal: (1/4) * Qᵢⱼ(1 - sᵢ - sⱼ + sᵢsⱼ)
                    offset += 0.25 * q_val
                    ising_dict[(i,)] = ising_dict.get((i,), 0.0) - 0.25 * q_val
                    ising_dict[(j,)] = ising_dict.get((j,), 0.0) - 0.25 * q_val
                    ising_dict[(i, j)] = ising_dict.get((i, j), 0.0) + 0.25 * q_val
    
    return ising_dict, offset


if __name__ == "__main__":
    # Example usage
    print("=" * 70)
    print("QUBO FORMULATION TEST")
    print("=" * 70)
    
    # Create sample portfolio
    np.random.seed(42)
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA']
    n_stocks = len(tickers)
    n_days = 252
    
    # Generate sample returns
    mean_returns = np.array([0.25, 0.22, 0.20, 0.35, 0.15]) / 252
    volatilities = np.array([0.30, 0.25, 0.28, 0.45, 0.60]) / np.sqrt(252)
    
    corr = np.array([
        [1.00, 0.75, 0.70, 0.65, 0.50],
        [0.75, 1.00, 0.80, 0.70, 0.45],
        [0.70, 0.80, 1.00, 0.68, 0.42],
        [0.65, 0.70, 0.68, 1.00, 0.55],
        [0.50, 0.45, 0.42, 0.55, 1.00],
    ])
    
    cov = np.outer(volatilities, volatilities) * corr
    returns_array = np.random.multivariate_normal(mean_returns, cov, n_days)
    returns_df = pd.DataFrame(returns_array, columns=tickers)
    
    # Formulate QUBO
    print(f"\nPortfolio: {', '.join(tickers)}")
    print(f"Budget: Select 3 stocks")
    
    qubo = PortfolioQUBO(risk_factor=1.0, return_factor=0.5, budget_penalty=10.0)
    Q = qubo.formulate(returns_df, budget=3)
    
    print(f"\nQUBO Matrix shape: {Q.shape}")
    print(f"QUBO Matrix:\n{Q}")
    
    # Test a few solutions
    print("\n" + "-" * 70)
    print("TESTING SOLUTIONS")
    print("-" * 70)
    
    test_solutions = [
        "11100",  # Select first 3
        "10101",  # Select 0, 2, 4
        "01110",  # Select 1, 2, 3
    ]
    
    for solution in test_solutions:
        selected_indices, selected, weights = qubo.decode_solution(solution)
        objective = qubo.evaluate_objective(solution)
        print(f"\nSolution: {solution}")
        print(f"  Selected: {', '.join(selected)}")
        print(f"  Weights: {weights}")
        print(f"  Objective: {objective:.4f}")
    
    # Test Ising conversion
    print("\n" + "-" * 70)
    print("ISING CONVERSION")
    print("-" * 70)
    
    ising_dict, offset = qubo_to_ising(Q)
    print(f"\nIsing coefficients: {len(ising_dict)} terms")
    print(f"Offset: {offset:.4f}")
    print(f"\nSample Ising terms:")
    for i, (key, val) in enumerate(list(ising_dict.items())[:5]):
        print(f"  {key}: {val:.4f}")
    
    print("\n" + "=" * 70)
    print("QUBO FORMULATION TEST COMPLETE")
    print("=" * 70)
