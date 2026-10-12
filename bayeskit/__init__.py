"""Empiricist Geophysics toolkit: small problems, readable samplers, diagnostics and the house plot style.

    import bayeskit as bk
    prob = bk.problems.location(kind="banana")
    res  = bk.samplers.nuts(prob, x0=[0, 0], n=2000)
"""
from . import diagnostics, plotting, problems, samplers, style  # noqa: F401
from .style import COLORS, savefig, show_gif  # noqa: F401

style.use()
