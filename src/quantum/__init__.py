"""
Quantum Portfolio Optimization Module
Implements QAOA for portfolio optimization using Qiskit
"""

try:
    from .qubo_formulation import PortfolioQUBO, qubo_to_ising
    from .qaoa_optimizer import QAOAOptimizer
    QISKIT_AVAILABLE = True
    __all__ = ['PortfolioQUBO', 'qubo_to_ising', 'QAOAOptimizer']
except (ImportError, Exception) as _e:
    QISKIT_AVAILABLE = False
    __all__ = []
