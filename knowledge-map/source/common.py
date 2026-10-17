"""Helpers for writing the knowledge map by hand. Each topic is written with T(...); prerequisites are listed inside
the topic that needs them, as (prerequisite_id, "hard" | "soft", reason)."""

SUBJECT = "Inverse problems and Bayesian inversion"          # Topic 2 of the course

DOMAINS = {
    "A": "A · Probability for inverse problems",
    "B": "B · Linear inversion, ill-posedness, priors",
    "C": "C · Errors and data weighting",
    "D": "D · Computing a posterior: sampling",
    "E": "E · Time-dependent and ensemble inversion",
    "F": "F · Hyperparameters and weights",
    "G": "G · Checking and synthesis",
}

TYPES = ("CONCEPTUAL", "PROCEDURAL", "REPRESENTATIONAL", "LANGUAGE", "META")


def T(id, type, dom, nb, sec, name, desc, evidence, ask, pre=(), fig=None, alias=()):
    """One micro-topic.
    id        stable identifier, "eg_" + snake_case
    type      CONCEPTUAL (an idea), PROCEDURAL (a method you carry out), REPRESENTATIONAL (a picture or representation
              you learn to read), LANGUAGE (a vocabulary distinction), META (a judgement: checking, choosing, reporting)
    dom       section letter A-G
    nb, sec   notebook id ("2.06") and section number where the topic is introduced (None: no numbered section)
    pre       prerequisites: (id, "hard" | "soft", one-line reason)
    fig       key figure in figures/ (optional)
    alias     words used to find the topic in the notebooks (the name is always included)
    """
    return dict(id=id, type=type, domain=DOMAINS[dom], introducedIn=nb, section=sec, name=name, description=desc,
                evidence=list(evidence), assessmentPrompt=ask, prerequisites=[tuple(p) for p in pre],
                figure=fig, aliases=list(alias))
