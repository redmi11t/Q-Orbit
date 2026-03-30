"""
IBM Quantum Backend Helper for Q-Orbit
Provides a thin wrapper around qiskit-ibm-runtime to connect to real IBM QPUs.
Falls back gracefully when token is not set or connection fails.
"""

import os
import warnings
from typing import Optional


class IBMBackendHelper:
    """
    Manages IBM Quantum Runtime authentication and backend selection.

    Usage:
        helper = IBMBackendHelper()
        backend = helper.get_backend()   # None if not configured / failed
        if backend is None:
            print(helper.last_error)     # See the real reason
    """

    def __init__(
        self,
        token: Optional[str] = None,
        backend_name: Optional[str] = None,
        instance: str = "ibm-q/open/main",
    ):
        self.token = token or os.getenv("IBM_QUANTUM_TOKEN", "")
        self.backend_name = (
            backend_name
            or os.getenv("IBM_QUANTUM_BACKEND", "ibm_brisbane")
        )
        self.instance = instance
        self._service = None
        self._backend = None
        self.last_error: Optional[str] = None   # Always set on failure

    # ------------------------------------------------------------------
    def is_configured(self) -> bool:
        return bool(self.token and self.token.strip())

    # ------------------------------------------------------------------
    def get_backend(self, timeout: int = 15):
        """
        Authenticate and return the IBM Quantum backend object.
        Tries 'ibm_quantum' channel first, falls back to 'ibm_cloud'.

        Returns backend object or None (check self.last_error for reason).
        """
        if not self.is_configured():
            self.last_error = (
                "IBM_QUANTUM_TOKEN is not set in .env. "
                "Add your token to use real IBM hardware."
            )
            warnings.warn(self.last_error, RuntimeWarning, stacklevel=2)
            return None

        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
        except ImportError:
            self.last_error = (
                "qiskit-ibm-runtime is not installed. "
                "Run: pip install qiskit-ibm-runtime>=0.20.0"
            )
            warnings.warn(self.last_error, RuntimeWarning, stacklevel=2)
            return None

        # Try ibm_quantum channel (works for most API keys)
        for channel in ("ibm_quantum", "ibm_cloud"):
            try:
                service = QiskitRuntimeService(
                    channel=channel,
                    token=self.token,
                )
                # List backends — pick exact name or closest match
                available = service.backends()
                available_names = [b.name for b in available]

                # Find requested backend
                if self.backend_name in available_names:
                    backend = service.backend(self.backend_name)
                else:
                    # Try to find any operational simulator as fallback
                    sim_names = [n for n in available_names if "simulator" in n.lower()]
                    fallback_name = sim_names[0] if sim_names else (available_names[0] if available_names else None)
                    if fallback_name is None:
                        self.last_error = (
                            f"[{channel}] Connected but no backends found. "
                            f"Your account may have no active instances."
                        )
                        continue
                    self.last_error = (
                        f"Backend '{self.backend_name}' not found on your account. "
                        f"Available: {available_names}. "
                        f"Auto-selected: '{fallback_name}'."
                    )
                    warnings.warn(self.last_error, RuntimeWarning, stacklevel=2)
                    backend = service.backend(fallback_name)
                    self.backend_name = fallback_name

                self._service = service
                self._backend = backend
                self.last_error = None
                return backend

            except Exception as exc:
                self.last_error = f"[{channel}] {type(exc).__name__}: {exc}"
                continue  # try next channel

        # Both channels failed
        warnings.warn(
            f"IBM Quantum connection failed: {self.last_error}",
            RuntimeWarning, stacklevel=2,
        )
        return None

    # ------------------------------------------------------------------
    def list_available_backends(self) -> list:
        """Quick list of backend names — returns [] on failure."""
        if not self.is_configured():
            return []
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
            for channel in ("ibm_quantum", "ibm_cloud"):
                try:
                    service = QiskitRuntimeService(channel=channel, token=self.token)
                    return [b.name for b in service.backends()]
                except Exception:
                    continue
        except Exception:
            pass
        return []

    # ------------------------------------------------------------------
    def get_backend_status(self) -> dict:
        """Returns a status dict for the Streamlit sidebar."""
        if not self.is_configured():
            return {
                "configured": False,
                "backend_name": self.backend_name,
                "available": False,
                "error": "No IBM_QUANTUM_TOKEN set in .env",
            }
        backend = self.get_backend()
        if backend is None:
            return {
                "configured": True,
                "backend_name": self.backend_name,
                "available": False,
                "error": self.last_error,
            }
        try:
            status = backend.status()
            return {
                "configured": True,
                "backend_name": self.backend_name,
                "available": status.operational,
                "pending_jobs": getattr(status, "pending_jobs", "N/A"),
                "error": None,
            }
        except Exception as exc:
            return {
                "configured": True,
                "backend_name": self.backend_name,
                "available": False,
                "error": str(exc),
            }


# ------------------------------------------------------------------
if __name__ == "__main__":
    helper = IBMBackendHelper()
    print(f"Configured: {helper.is_configured()}")
    if helper.is_configured():
        print(f"Connecting to: {helper.backend_name} ...")
        b = helper.get_backend()
        if b:
            print(f"Connected to: {b.name}")
        else:
            print(f"Failed: {helper.last_error}")
