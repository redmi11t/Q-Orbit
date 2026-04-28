"""
Classical Portfolio Optimization
Baseline Markowitz Mean-Variance optimization using modern techniques
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.optimize import minimize, differential_evolution, LinearConstraint
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
        Find the portfolio with maximum Sharpe ratio.

        Uses a two-stage approach:
          1. ``differential_evolution`` (global) to escape local minima.
          2. Multi-start SLSQP (local) to polish the solution.

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
        # Use (A + A.T)/2 to get a writable, symmetric NumPy copy
        # (sigma.values alone returns a read-only view of the DataFrame internals)
        sigma_np = (sigma.values + sigma.values.T) / 2
        sigma_np += 1e-6 * np.eye(n_assets)
        
        # Objective function: negative Sharpe ratio (to minimize)
        def neg_sharpe(weights):
            port_return = np.dot(weights, mu)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(sigma_np, weights)))
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

        # ── Fix 5: Global search with differential_evolution ─────────────────
        # SLSQP is gradient-based and can converge to a local minimum.  We run
        # a global search first and use the best solution found as the primary
        # warm-start for SLSQP.  This adds a small amount of compute time but
        # substantially reduces the chance of suboptimal convergence.
        initial_guesses = []
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                de_result = differential_evolution(
                    neg_sharpe,
                    bounds=bounds,
                    maxiter=300,
                    tol=1e-8,
                    seed=42,
                    constraints=LinearConstraint(
                        np.ones((1, n_assets)), lb=1.0, ub=1.0
                    ),
                    polish=False,   # We polish with SLSQP below
                    workers=1,      # Keep deterministic / avoid pickling issues
                )
            if de_result.success or de_result.fun < 0:
                # Normalise to satisfy sum-to-1 exactly after DE
                de_x = np.clip(de_result.x, 0, max_weight)
                de_x = de_x / de_x.sum()
                initial_guesses.append(de_x)
        except Exception:
            pass  # Fall through to the deterministic guesses below
        # ─────────────────────────────────────────────────────────────────────

        # Deterministic fallback guesses (equal weight, dirichlet, biased)
        # Fix #5: Use a seeded RNG instead of the global numpy state so that
        # results are reproducible across reruns regardless of external RNG state.
        _rng = np.random.default_rng(42)
        initial_guesses.append(np.array([1/n_assets] * n_assets))
        initial_guesses.append(_rng.dirichlet(np.ones(n_assets)))
        
        # Add a guess biased toward the highest-return asset
        biased = np.ones(n_assets) * 0.5 / n_assets
        biased[np.argmax(mu)] += 0.5
        initial_guesses.append(biased)
        
        best_result = None
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

        # Phase 1 Fix #3: Guard against solver failure — w.value is None when
        # CVXPY fails (numerical issues, infeasible problem, etc.).
        # Propagating None silently would create an all-NaN weight Series.
        if problem.status not in ('optimal', 'optimal_inaccurate') or w.value is None:
            raise RuntimeError(
                f"Min-variance solver failed (status='{problem.status}'). "
                "Try using fewer stocks or enabling 'Use Real Market Data'."
            )

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
        # Fix #1: Add regularization (same as optimize_min_variance) to guarantee
        # positive-definiteness on near-singular real covariance matrices.
        sigma_np += 1e-6 * np.eye(n_assets)
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

        # Phase 1 Fix #4: Same NaN guard for optimize_target_return.
        # Called in a loop by generate_efficient_frontier — raise so the
        # caller can gracefully skip infeasible frontier points.
        if problem.status not in ('optimal', 'optimal_inaccurate') or w.value is None:
            raise RuntimeError(
                f"Target-return solver failed (status='{problem.status}', "
                f"target={target_return:.4f}). This target may be infeasible."
            )

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

        # Phase 2 Fix #11: Save current optimizer state so the loop below
        # doesn't corrupt self.weights / self.performance that the main app
        # stored in session_state.  A fresh temporary optimizer is used for
        # each frontier point; state is restored at the end regardless.
        _saved_weights = self.weights
        _saved_performance = self.performance
        try:
            tmp_opt = MarkowitzOptimizer(risk_free_rate=self.risk_free_rate)
            for target in target_returns:
                try:
                    tmp_opt.optimize_target_return(returns, target)
                    risks.append(tmp_opt.performance['annual_volatility'])
                    frontier_returns.append(tmp_opt.performance['annual_return'])
                    weights_list.append(tmp_opt.weights.values)
                except Exception as e:
                    # Some target returns might be infeasible, especially at the extremes
                    print(f"  Note: Could not solve for target return {target:.2%}: {e}")
                    continue
        finally:
            # Always restore original state
            self.weights = _saved_weights
            self.performance = _saved_performance

        return np.array(risks), np.array(frontier_returns), weights_list
    
    def _calculate_performance(self, returns: pd.DataFrame, weights: pd.Series):
        """Calculate portfolio performance metrics"""
        portfolio_returns = (returns @ weights)
        
        annual_return = portfolio_returns.mean() * 252
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        # Clamp volatility to avoid near-zero denominator (can happen with
        # synthetic / heavily-smoothed data), which would blow Sharpe to 300+.
        annual_volatility_safe = max(annual_volatility, 1e-6)
        sharpe_ratio = (annual_return - self.risk_free_rate) / annual_volatility_safe
        # Cap at ±50: any value outside this range is a numerical artifact
        sharpe_ratio = float(np.clip(sharpe_ratio, -50.0, 50.0))

        # Drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Sortino Ratio (penalises only downside/negative returns)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        if len(downside_returns) > 1:
            downside_std = downside_returns.std() * np.sqrt(252)
        else:
            # No downside returns → use total volatility as a conservative proxy
            # so we don't divide by ~0 and produce multi-billion Sortino values
            downside_std = annual_volatility_safe
        # Apply the same 1e-6 floor to downside_std as well
        downside_std = max(downside_std, 1e-6)
        sortino_ratio = (annual_return - self.risk_free_rate) / downside_std
        # Cap at ±50 for the same reason
        sortino_ratio = float(np.clip(sortino_ratio, -50.0, 50.0))

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
