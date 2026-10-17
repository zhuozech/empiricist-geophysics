"""Integrity check for the knowledge map. Exits non-zero on any error; warnings are printed but allowed.

Checks: declared counts; ids, types, domains and required fields; every edge endpoint exists; no self-loops or duplicate
edges; the prerequisite graph has no cycles; hard prerequisites respect the reading order (introduced in an earlier
notebook, or earlier or the same section of the same one); every notebook, section link and figure exists; manifest
SHA-256 checksums.

usage: python knowledge-map/scripts/validate.py
"""
import hashlib
import json
import pathlib
import re
import sys

KM = pathlib.Path(__file__).resolve().parent.parent
ROOT = KM.parent
DATA = KM / "data"
TYPES = {"CONCEPTUAL", "PROCEDURAL", "REPRESENTATIONAL", "LANGUAGE", "META"}
errors, warnings = [], []


def check(cond, msg):
    if not cond:
        errors.append(msg)


def load(name):
    return json.loads((DATA / name).read_text())


topics, deps, clusters, manifest = (load(n) for n in ("topics.json", "dependencies.json", "clusters.json", "manifest.json"))
T = topics["topics"]
D = deps["dependencies"]
check(topics["topicCount"] == len(T), f"topicCount {topics['topicCount']} != {len(T)}")
check(deps["edgeCount"] == len(D), f"edgeCount {deps['edgeCount']} != {len(D)}")
check(clusters["clusterCount"] == len(clusters["clusters"]), "clusterCount mismatch")

domains = {c["domain"] for c in clusters["clusters"]}
notebooks = {p.name[:4].replace("-", "."): p for p in (ROOT / "notebooks").glob("[0-9]-[0-9][0-9]_*.ipynb")}
byid = {}
for t in T:
    i = t.get("id", "")
    check(re.fullmatch(r"eg_[a-z0-9_]+", i or "") is not None, f"bad id {i!r}")
    check(i not in byid, f"duplicate id {i}")
    byid[i] = t
    check(t.get("type") in TYPES, f"{i}: bad type {t.get('type')}")
    check(t.get("domain") in domains, f"{i}: unknown domain {t.get('domain')}")
    for f in ("name", "description", "assessmentPrompt"):
        check(isinstance(t.get(f), str) and t[f].strip(), f"{i}: empty {f}")
    check(isinstance(t.get("evidence"), list) and len(t["evidence"]) >= 1, f"{i}: needs at least one evidence item")
    check(t.get("introducedIn") in notebooks, f"{i}: notebook {t.get('introducedIn')} not found")
    if t.get("figure"):
        check((ROOT / "figures" / t["figure"]).exists(), f"{i}: figure {t['figure']} not found")
    if t.get("url"):
        page, _, anchor = t["url"].partition("#")
        p = ROOT / page
        check(p.exists(), f"{i}: page {page} not found")
        if anchor and p.exists():
            check(f'id="{anchor}"' in p.read_text(errors="ignore"), f"{i}: anchor #{anchor} not found in {page}")
    for u in t.get("usedIn", []):
        check(u in notebooks, f"{i}: usedIn notebook {u} not found")
        check(float(u) > float(t["introducedIn"]), f"{i}: usedIn {u} is not after {t['introducedIn']}")
    for u in t.get("mentionedBefore", []):
        check(u in notebooks and float(u) < float(t["introducedIn"]), f"{i}: bad mentionedBefore {u}")
    c = t.get("centrality")
    check(c is None or 0 <= c <= 1, f"{i}: centrality {c} outside [0, 1]")

seen = set()
parents = {i: [] for i in byid}
order = lambda t: (float(t["introducedIn"]), t["section"] if t["section"] is not None else 0)
forward_soft = 0
for d in D:
    a, b = d["topicId"], d["prerequisiteId"]
    check(a in byid, f"edge topic {a} missing"); check(b in byid, f"edge prerequisite {b} missing")
    check(a != b, f"self-loop on {a}")
    check(d["strength"] in ("hard", "soft"), f"edge {b}->{a}: bad strength")
    check((a, b) not in seen, f"duplicate edge {b}->{a}")
    seen.add((a, b))
    if a in byid and b in byid:
        parents[a].append(b)
        if order(byid[b]) > order(byid[a]):
            if d["strength"] == "hard":
                errors.append(f"hard prerequisite {b} ({byid[b]['introducedIn']}) comes after {a} ({byid[a]['introducedIn']})")
            else:
                forward_soft += 1
                warnings.append(f"soft forward reference: {a} ({byid[a]['introducedIn']}) points ahead to {b} "
                                f"({byid[b]['introducedIn']})")

# cycle check (depth-first search)
state = {}


def visit(n, path):
    if state.get(n) == 1:
        errors.append("cycle: " + " -> ".join(path + [n])); return
    if state.get(n) == 2:
        return
    state[n] = 1
    for p in parents.get(n, []):
        visit(p, path + [n])
    state[n] = 2


for n in byid:
    visit(n, [])

for name, info in manifest["files"].items():
    raw = (DATA / name).read_bytes()
    check(hashlib.sha256(raw).hexdigest() == info["sha256"], f"{name}: checksum mismatch")
    check(len(raw) == info["bytes"], f"{name}: size mismatch")

isolated = [i for i in byid if not parents[i] and not any(d["prerequisiteId"] == i for d in D)]
for i in isolated:
    warnings.append(f"isolated topic (no edges): {i}")
for w in warnings:
    print("warning:", w)
print(f"{len(T)} topics, {len(D)} edges ({sum(d['strength'] == 'hard' for d in D)} hard), "
      f"{forward_soft} soft forward references, {len(isolated)} isolated topics")
if errors:
    for e in errors:
        print("ERROR:", e)
    print(f"FAILED with {len(errors)} errors")
    sys.exit(1)
print("OK: all checks passed")
