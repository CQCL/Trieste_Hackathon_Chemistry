from numpy import ndarray
from typing import Tuple
import numpy
import scipy

def gen_eigh(
    h: ndarray, s: ndarray, lindep: float = 1e-14
) -> Tuple[ndarray, ndarray, ndarray]:
    """Solves generalized eigenvalue problem :math:`HC = SCE`.
    Args:
        h (array-like): Hermitian matrix.
        s (array-like): Hermitian matrix.
        lindep: Tolerance to determine linear dependency.
    Return:
        e: Eigenvalues in the linear-independent subspace.
        c: Eigenvectors.
        es: Eigenvalues of `s`.
    """
    # NOTE: see also pyscf.lib.linalg_helper.safe_eigh.
    # Diagonalize S (overlap).
    es, cs = scipy.linalg.eigh(s)
    # Calculate the linear-independent subspace (canonical space).
    for i, esi in enumerate(es):
        if numpy.abs(esi) >= lindep and esi > 0.0:
            nsub = i
            break
    cs_sub = cs[:, nsub:]
    # The X matrix (canonicalization matrix).
    x = cs_sub.dot(numpy.diag(es[nsub:] ** -0.5))
    # Diagonalize the X^HX matrix.
    xthx = numpy.conj(x.T).dot(h).dot(x)
    e, c = scipy.linalg.eigh(xthx)
    # Transform C back to the original space.
    c = x.dot(c)
    return e, c, es