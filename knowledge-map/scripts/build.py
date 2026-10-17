"""Build the knowledge map of Topic 2 from knowledge-map/source/ into knowledge-map/data/*.json.

Computed fields (derived, not hand-written):
  usedIn           later notebooks whose text mentions the topic (by its specific aliases)
  mentionedBefore  earlier notebooks that mention it before it is introduced (previews that should point ahead)
  centrality  share of all topics that need this topic directly or indirectly, scaled so the largest is 1
  url         link to the introducing notebook section in html/

usage: python knowledge-map/scripts/build.py
"""
from __future__ import annotations

import hashlib
import html as H
import json
import pathlib
import re
import sys
from collections import defaultdict

KM = pathlib.Path(__file__).resolve().parent.parent
ROOT = KM.parent
sys.path.insert(0, str(KM / "source"))
from common import DOMAINS, SUBJECT  # noqa: E402
from clusters import CLUSTERS  # noqa: E402
import part_a_b, part_c_d, part_e_g  # noqa: E402

VERSION = "v1"
# aliases matched case-sensitively in their usual spelling (acronyms)
ACRONYMS = {"map": "MAP", "ess": "ESS", "smc": "SMC", "nuts": "NUTS", "hmc": "HMC", "jax": "JAX", "ekf": "EKF",
            "enki": "EnKI", "enkf": "EnKF", "eki": "EKI", "gp": "GP", "sbc": "SBC", "loo": "LOO", "waic": "WAIC",
            "aic": "AIC", "bic": "BIC", "abic": "ABIC", "bur": "BUR", "br": "BR", "svd": "SVD", "tv": "TV", "l1": "L1",
            "l2": "L2", "rhat": "R-hat", "r-hat": "R-hat", "mcse": "MCSE", "rts": "RTS", "hpd": "HPD", "gcv": "GCV",
            "catmip": "CATMIP", "wrms": "WRMS", "pdf": "pdf", "cdf": "cdf", "elbo": "ELBO", "jit": "jit", "vmap": "vmap"}
# words too common to say "this topic is used here" (still used to check the introducing section)
STOP = {"shape", "mean", "mode", "median", "noise", "prior", "posterior", "gradient", "update", "predict", "filter",
        "cost", "draws", "samples", "flat", "smooth", "band", "grid", "sampler", "inversion", "bias", "interval",
        "joint", "marginal", "conditional", "positive", "wrong", "ranges", "ridge", "monitor", "seed", "cap",
        "transform", "bounds", "scaling", "regression", "information", "evidence", "misfit", "sensitivity",
        "reporting", "averaging", "overlap", "checklist", "tails", "uniform", "nonlinear", "linearisation",
        "linearization", "profile", "degrade", "rounded", "terraces", "adaptive", "schedule", "stiff", "nan", "flow",
        "span", "subspace", "truth", "dimension", "multistart", "notation", "residual", "gain", "inverse", "rank",
        "precision", "ensemble", "members", "sequential", "proposal", "acceptance", "crowd", "objective", "mixing",
        "convergence", "converged", "θ", "theta", "β", "λ", "jitter", "pivot", "draws", "independent", "correlation",
        "covariance", "variance", "expectation", "likelihood", "shrinkage", "sparse", "map", "bit", "gnss", "tsunami",
        "insar", "flicker", "random walk", "random-walk", "white noise", "model error", "c_p", "c_d", "positivity",
        "hierarchical", "hierarchy", "hyperparameter", "regularised", "unregularised", "marginal likelihood", "kalman filter",
        "gaussian", "laplace", "student-t", "after sampling", "momentum", "potential energy", "gaussian posterior",
        "linear forward model", "maximum likelihood", "noise units", "units of the data noise", "two islands",
        "patch by patch", "what the data teach", "a minimum is not an uncertainty", "every choice", "capstone",
        "two-dimensional", "fault plane", "black box", "black-box", "differentiable", "optimiser", "optimizer",
        "trace plot", "tail", "bulk", "particles", "annealing", "tempering", "elbo", "flat prior", "simultaneous band",
        "mass-covering", "mode-seeking", "asymmetry", "interval", "derived quantity", "cross-covariance", "smoothing",
        "gp", "eki", "br", "tv", "l1", "l2", "jit", "pdf", "cdf", "regularisation", "regularization", "regulariser",
        "penalty", "laplacian", "filtering", "periodogram", "nonlinearity", "non-gaussian", "non-gaussianity",
        "noise scale", "data weight", "degree of nonlinearity", "information gain", "full covariance", "independent data",
        "effective number", "decimation", "covariogram", "bounds", "dimension", "ridge", "sequential bayes", "kernel",
        "length scale", "amplitude", "regression", "proposal", "acceptance rate", "step size", "hessian", "curvature"}


def load_topics():
    return part_a_b.TOPICS + part_c_d.TOPICS + part_e_g.TOPICS


def notebook_sources():
    out = {}
    for p in sorted((ROOT / "build" / "src").glob("2-*.py")):
        out[p.name[:4].replace("-", ".")] = p
    return out


def markdown_sections(path):
    """{section number or None: markdown text}; None holds the whole notebook's markdown."""
    secs, cur, whole = defaultdict(list), "pre", []
    for line in path.read_text().splitlines():
        if not line.startswith("#") or line.startswith("# %%"):
            continue
        txt = line[2:] if line.startswith("# ") else line.lstrip("#")
        m = re.match(r"## (\d+)(?:-(\d+))?\.", txt)
        if m:
            cur = int(m.group(1))
        elif txt.startswith("## "):
            cur = "other"
        secs[cur].append(txt); whole.append(txt)
        if m and m.group(2):                                   # "## 3-4." heading covers both numbers
            secs[int(m.group(2))].append(txt)
    out = {k: "\n".join(v) for k, v in secs.items()}
    out[None] = "\n".join(whole)
    return out


def finder(alias):
    a = alias.strip()
    canon = ACRONYMS.get(a.lower())
    if canon:
        return re.compile(r"(?<![\w-])" + re.escape(canon) + r"(?![\w])")
    return re.compile(r"(?<![\w])" + re.escape(a) + r"(?![\w])", re.I)


def section_anchors():
    """{notebook id: {section number: html anchor}} read from the built pages."""
    out = {}
    for p in sorted((ROOT / "html").glob("2-*.html")):
        ids = re.findall(r'<h[23] id="([^"]+)"', p.read_text(errors="ignore"))
        d = {}
        for i in ids:
            m = re.match(r"(\d+)(?:-\d+)?\.", i)
            if m and int(m.group(1)) not in d:
                d[int(m.group(1))] = i
        out[p.name[:4].replace("-", ".")] = (p.name, d)
    return out


def main():
    topics = load_topics()
    nbs = notebook_sources()
    secs = {k: markdown_sections(v) for k, v in nbs.items()}
    anchors = section_anchors()
    ids = [t["id"] for t in topics]
    byid = {t["id"]: t for t in topics}
    warnings = []

    # dependencies (each topic lists its prerequisites; the data file stores directed edges)
    deps = []
    for t in topics:
        for pid, strength, reason in t["prerequisites"]:
            deps.append(dict(topicId=t["id"], prerequisiteId=pid, strength=strength, reason=reason))

    # centrality: share of topics that need this topic (directly or indirectly)
    children = defaultdict(set)
    for d in deps:
        children[d["prerequisiteId"]].add(d["topicId"])

    def descendants(x, seen=None):
        seen = set() if seen is None else seen
        for c in children[x]:
            if c not in seen:
                seen.add(c); descendants(c, seen)
        return seen
    ndesc = {i: len(descendants(i)) for i in ids}
    top = max(ndesc.values()) or 1

    out_topics = []
    for t in topics:
        nb = t["introducedIn"]
        als = [t["name"]] + t["aliases"]
        sec_text = secs.get(nb, {}).get(t["section"], "") if t["section"] is not None else secs.get(nb, {}).get(None, "")
        if not any(finder(a).search(sec_text) for a in als):
            warnings.append(f"{t['id']}: none of its words found in {nb} section {t['section']}")
        specific = [a for a in t["aliases"] if a.lower() not in STOP and (len(a) >= 4 or a.lower() in ACRONYMS)]
        hits = sorted(k for k, s in secs.items() if k != nb and specific
                      and any(finder(a).search(s[None]) for a in specific))
        used = [k for k in hits if float(k) > float(nb)]
        before = [k for k in hits if float(k) < float(nb)]
        page, amap = anchors.get(nb, (None, {}))
        url = None
        if page:
            url = f"html/{page}" + (f"#{amap[t['section']]}" if t["section"] in amap else "")
        out_topics.append(dict(id=t["id"], type=t["type"], subject=SUBJECT, domain=t["domain"], name=t["name"],
                               description=t["description"], introducedIn=nb, section=t["section"], url=url,
                               usedIn=used, mentionedBefore=before, figure=t["figure"], centrality=round(ndesc[t["id"]] / top, 4),
                               evidence=t["evidence"], assessmentPrompt=t["assessmentPrompt"]))

    data = KM / "data"
    data.mkdir(exist_ok=True)
    files = {
        "topics.json": dict(version=VERSION, topicCount=len(out_topics), topics=out_topics),
        "dependencies.json": dict(version=VERSION, note="`topicId` depends on `prerequisiteId` (the prerequisite). "
                                  "Directed edges of a DAG.", edgeCount=len(deps), dependencies=deps),
        "clusters.json": dict(version=VERSION, clusterCount=len(CLUSTERS), clusters=[
            dict(subject=SUBJECT, domain=DOMAINS[k], firstNotebook=min(t["introducedIn"] for t in topics
                                                                         if t["domain"] == DOMAINS[k]),
                 lastNotebook=max(t["introducedIn"] for t in topics if t["domain"] == DOMAINS[k]),
                 topicCount=sum(t["domain"] == DOMAINS[k] for t in topics), summary=CLUSTERS[k])
            for k in DOMAINS]),
    }
    for name, obj in files.items():
        (data / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")
    by_type, by_dom = defaultdict(int), defaultdict(int)
    for t in out_topics:
        by_type[t["type"]] += 1; by_dom[t["domain"]] += 1
    manifest = dict(dataset="Empiricist Geophysics knowledge map", taxonomyVersion=VERSION, subject=SUBJECT,
                    counts=dict(topics=len(out_topics), topicsByDomain=dict(by_dom), topicsByType=dict(by_type),
                                dependencies=len(deps), hard=sum(d["strength"] == "hard" for d in deps),
                                soft=sum(d["strength"] == "soft" for d in deps), clusters=len(CLUSTERS),
                                notebooks=len({t["introducedIn"] for t in out_topics})),
                    files={n: dict(bytes=(data / n).stat().st_size,
                                   sha256=hashlib.sha256((data / n).read_bytes()).hexdigest()) for n in files})
    (data / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    c = manifest["counts"]
    order = ["CONCEPTUAL", "PROCEDURAL", "REPRESENTATIONAL", "LANGUAGE", "META"]
    block = (f"**{c['topics']} topics** from {c['notebooks']} notebooks, **{c['dependencies']} prerequisite links** "
             f"({c['hard']} hard, {c['soft']} soft), {c['clusters']} sections.\n\n| section | topics |\n|---|---|\n"
             + "".join(f"| {k} | {v} |\n" for k, v in c["topicsByDomain"].items())
             + "\nBy type: " + ", ".join(f"{t.lower()} {c['topicsByType'].get(t, 0)}" for t in order) + ".\n")
    rp = KM / "README.md"; r = rp.read_text()
    i, j = r.index("<!-- counts -->"), r.index("<!-- /counts -->")
    rp.write_text(r[:i] + "<!-- counts -->\n" + block + r[j:])
    print(f"wrote {len(out_topics)} topics, {len(deps)} dependencies, {len(CLUSTERS)} clusters to {data}")
    for w in warnings:
        print("WARNING", w)
    print(f"{len(warnings)} warnings")


if __name__ == "__main__":
    main()
