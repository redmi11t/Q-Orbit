"""
Classical Portfolio Optimization
Baseline Markowitz Mean-Variance optimization using modern techniques
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.optimize import minimize
import cvxpy as cp


class MarkowitzOptimizer:
    """Classical Mean-Variance Portfolio Optimization"""
    
    def __init__(self, risk_free_rate: float = 0.04):
        """
        Initialize optimizer
        
        Args:
            risk_free_rate: Annual risk-free rate (default: 4%)
        """
        self.risk_free_rate = risk_free_rate
        self.weights = None
        self.performance = None
        
    def optimize_max_sharpe(
        self,
        returns: pd.DataFrame,
        constraints: Optional[Dict] = None
    ) -> pd.Series:
        """
        Find the portfolio with maximum Sharpe ratio using scipy
        
        Args:
            returns: DataFrame of asset returns
            constraints: Optional constraints dict
            
        Returns:
            Series of optimal weights
        """
        n_assets = len(returns.columns)
        
        # Calculate expected returns and covariance
        mu = returns.mean() * 252  # Annualized
        sigma = returns.cov() * 252
        # Add regularization to covariance matrix
        sigma.values += 1e-6 * np.eye(n_assets)
        
        # Objective function: negative Sharpe ratio (to minimize)
        def neg_sharpe(weights):
            port_return = np.dot(weights, mu)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))
            if port_vol < 1e-10:
                return 0.0
            return -(port_return - self.risk_free_rate) / port_vol
        
        # Bounds: all weights between 0 and 1 (no short selling)
        max_weight = 1.0
        if constraints and 'max_weight' in constraints:
            max_weight = constraints['max_weight']
        
        # Ensure bounds are feasible: max_weight * n_assets must be >= 1
        if max_weight * n_assets < 1.0:
            max_weight = 1.0 / n_assets + 0.01
        
        bounds = tuple((0, max_weight) for _ in range(n_assets))
        
        # Constraints: weights sum to 1
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        # Try multiple initial guesses and methods for robustness
        best_result = None
        initial_guesses = [
            np.array([1/n_assets] * n_assets),
            np.random.dirichlet(np.ones(n_assets)),
        ]
        
        # Add a guess biased toward the highest-return asset
        biased = np.ones(n_assets) * 0.5 / n_assets
        biased[np.argmax(mu)] += 0.5
        initial_guesses.append(biased)
        
        for x0 in initial_guesses:
            # Clip to bounds and re-normalize
            x0 = np.clip(x0, 0, max_weight)
            x0 = x0 / x0.sum()
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = minimize(
                        neg_sharpe,
                        x0,
                        method='SLSQP',
                        bounds=bounds,
                        constraints=cons,
                        options={'maxiter': 1000, 'ftol': 1e-9}
                    )
                    if result.success:
                        if best_result is None or result.fun < best_result.fun:
                            best_result = result
            except Exception:
                continue
        
        if best_result is not None:
            self.weights = pd.Series(best_result.x, index=returns.columns)
            self._calculate_performance(returns, self.weights)
            return self.weights
        
        # Fallback: use min-variance if max-Sharpe fails entirely
        warnings.warn("Max Sharpe optimization failed, falling back to minimum variance.")
        return self.optimize_min_variance(returns)
    
    def optimize_min_variance(
        self,
        returns: pd.DataFrame,
        constraints: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Find the minimum variance portfolio
        
        Args:
            returns: DataFrame of asset returns
            constraints: Optional constraints dict
            
        Returns:
            Array of optimal weights
        """
        n_assets = len(returns.columns)
        sigma = returns.cov() * 252
        # Enforce exact symmetry at the NumPy level to satisfy cvxpy's check
        sigma_np = (sigma.values + sigma.values.T) / 2
        # Add regularization to ensure positive definiteness
        sigma_np += 1e-6 * np.eye(n_assets)
        sigma_psd = cp.psd_wrap(sigma_np)
        
        w = cp.Variable(n_assets)
        
        # Objective: minimize variance
        objective = cp.Minimize(cp.quad_form(w, sigma_psd))
        
        # Constraints
        constraint_list = [
            cp.sum(w) == 1,
            w >= 0
        ]
        
        if constraints:
            if 'max_weight' in constraints:
                constraint_list.append(w <= constraints['max_weight'])
        
        problem = cp.Problem(objective, constraint_list)
        problem.solve()
        
        self.weights = pd.Series(w.value, index=returns.columns)
        self._calculate_performance(returns, self.weights)
        
        return self.weights
    
    def optimize_target_return(
        self,
        returns: pd.DataFrame,
        target_return: float,
        constraints: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Find portfolio with specific target return and minimum risk
        
        Args:
            returns: DataFrame of asset returns
            target_return: Desired annual return
            constraints: Optional constraints dict
            
        Returns:
            Array of optimal weights
        """
        n_assets = len(returns.columns)
        mu = returns.mean() * 252
        sigma = returns.cov() * 252
        # Enforce exact symmetry at the NumPy level to satisfy cvxpy's check
        sigma_np = (sigma.values + sigma.values.T) / 2
        sigma_psd = cp.psd_wrap(sigma_np)
        
        w = cp.Variable(n_assets)
        
        # Objective: minimize risk
        objective = cp.Minimize(cp.quad_form(w, sigma_psd))
        
        # Constraints
        constraint_list = [
            cp.sum(w) == 1,
            w >= 0,
            mu.values @ w >= target_return  # Target return constraint
        ]
        
        if constraints:
            if 'max_weight' in constraints:
                constraint_list.append(w <= constraints['max_weight'])
        
        problem = cp.Problem(objective, constraint_list)
        problem.solve()
        
        self.weights = pd.Series(w.value, index=returns.columns)
        self._calculate_performance(returns, self.weights)
        
        return self.weights.values
    
    def generate_efficient_frontier(
        self,
        returns: pd.DataFrame,
        n_points: int = 50
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Generate the efficient frontier
        
        Args:
            returns: DataFrame of asset returns
            n_points: Number of points on the frontier
            
        Returns:
            Tuple of (risks, returns, weights_list)
        """
        mu = returns.mean() * 252
        min_return = mu.min()
        max_return = mu.max()
        
        target_returns = np.linspace(min_return, max_return, n_points)
        
        risks = []
        frontier_returns = []
        weights_list = []
        
        for target in target_returns:
            try:
                weights = self.optimize_target_return(returns, target)
                risks.append(self.performance['annual_volatility'])
                frontier_returns.append(self.performance['annual_return'])
                weights_list.append(weights)
            except Exception as e:
                # Some target returns might be unfeasible, especially at the extremes
                print(f"  Note: Could not solve for target return {target:.2%}: {e}")
                continue
        
        return np.array(risks), np.array(frontier_returns), weights_list
    
    def _calculate_performance(self, returns: pd.DataFrame, weights: pd.Series):
        """Calculate portfolio performance metrics"""
        portfolio_returns = (returns @ weights)
        
        annual_return = portfolio_returns.mean() * 252
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_return - self.risk_free_rate) / annual_volatility
        
        # Drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Sortino Ratio (penalises only downside/negative returns)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 1e-10
        sortino_ratio = (annual_return - self.risk_free_rate) / downside_std

        # Calmar Ratio (annualised return / absolute max drawdown)
        calmar_ratio = annual_return / abs(max_drawdown) if abs(max_drawdown) > 1e-10 else 0.0
        
        self.performance = {
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'weights': weights
        }
    
    def get_performance_summary(self) -> Dict:
        """Get formatted performance summary"""
        if self.performance is None:
            return {}
        
        return {
            'Expected Return': f"{self.performance['annual_return']:.2%}",
            'Volatility': f"{self.performance['annual_volatility']:.2%}",
            'Sharpe Ratio': f"{self.performance['sharpe_ratio']:.3f}",
            'Sortino Ratio': f"{self.performance['sortino_ratio']:.3f}",
            'Calmar Ratio': f"{self.performance['calmar_ratio']:.3f}",
            'Max Drawdown': f"{self.performance['max_drawdown']:.2%}"
        }


if __name__ == "__main__":
    # Example usage
    from utils.data_loader import DataLoader, get_sample_portfolio
    
    # Load data
    loader = DataLoader()
    portfolio = get_sample_portfolio()
    prices = loader.fetch_price_data(
        portfolio['tickers'],
        portfolio['start_date'],
        portfolio['end_date']
    )
    returns = loader.calculate_returns(prices)
    
    # Optimize
    optimizer = MarkowitzOptimizer()
    
    print("=" * 60)
    print("MAX SHARPE RATIO PORTFOLIO")
    print("=" * 60)
    weights = optimizer.optimize_max_sharpe(returns)
    
    print("\nOptimal Weights:")
    for ticker, weight in optimizer.weights.items():
        if weight > 0.01:  # Only show significant weights
            print(f"  {ticker}: {weight:.2%}")
    
    print("\nPerformance:")
    for metric, value in optimizer.get_performance_summary().items():
        print(f"  {metric}: {value}")
