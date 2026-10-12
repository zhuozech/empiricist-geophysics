"""Turn notebook ids in the "Before this" and "Next" lines into links, and write html/index.html (the course index).

Runs on the executed notebooks; nothing is re-executed. Links point to .ipynb files inside notebooks/ and to .html
files inside html/, so both folders can be browsed on their own (the html/ folder works as a static website).

usage: python build/link_pages.py
"""
from __future__ import annotations

import html as H
import pathlib
import re

import nbformat
from nbconvert import HTMLExporter

ROOT = pathlib.Path(__file__).resolve().parent.parent
NB, HTML = ROOT / "notebooks", ROOT / "html"
STEMS = {p.stem[:4].replace("-", "."): p.stem for p in sorted(NB.glob("[0-9]-[0-9][0-9]_*.ipynb"))}
ID = re.compile(r"(?<![\w.\[])([23]\.\d\d)(?![\d\]])")


def _link_ids(text, ext):
    return ID.sub(lambda m: f"[{m.group(1)}]({STEMS[m.group(1)]}{ext})" if m.group(1) in STEMS else m.group(1), text)


def linkify(src, ext):
    """Link ids in the 'Before this' part of the header and in 'Next:' lines (idempotent)."""
    out = []
    for line in src.split("\n"):
        if "](" in line and (".ipynb)" in line or ".html)" in line):            # already linked
            out.append(line)
        elif line.startswith("Next:"):
            out.append(re.sub(r"\*\*([23]\.\d\d) · ([^*]+)\*\*",
                              lambda m: f"[**{m.group(1)} · {m.group(2)}**]({STEMS.get(m.group(1), '')}{ext})"
                              if m.group(1) in STEMS else m.group(0), line))
        elif "**Before this:**" in line:
            head, _, rest = line.partition("**Before this:**")
            before, sep, after = rest.partition("**You will be able to:**")
            out.append(head + "**Before this:**" + _link_ids(before, ext) + sep + after)
        else:
            out.append(line)
    return "\n".join(out)


def unlink(src):
    """Undo earlier links so the step can be rerun with another extension."""
    return re.sub(r"\[(\*\*[23]\.\d\d · [^*]+\*\*|[23]\.\d\d)\]\([\w.-]+\.(?:ipynb|html)\)", r"\1", src)


def first_heading(nb):
    for c in nb.cells:
        if c.cell_type == "markdown":
            m = re.search(r"^# (.+)$", c.source, re.M)
            if m:
                return m.group(1).strip()
    return ""


def main():
    exporter = HTMLExporter(template_name="lab")
    rows = []
    for key, stem in STEMS.items():
        path = NB / f"{stem}.ipynb"
        nb = nbformat.read(path, as_version=4)
        for c in nb.cells:
            if c.cell_type == "markdown":
                c.source = linkify(unlink(c.source), ".ipynb")
        nbformat.write(nb, path)
        for c in nb.cells:
            if c.cell_type == "markdown":
                c.source = linkify(unlink(c.source), ".html")
        body, _ = exporter.from_notebook_node(nb)
        (HTML / f"{stem}.html").write_text(body)
        rows.append((key, stem, first_heading(nb)))
    write_index(rows)
    print(f"linked {len(rows)} notebooks and wrote html/index.html")


def write_index(rows):
    sections = {"2.01": "A · Probability for inverse problems", "2.07": "B · Linear inversion, ill-posedness, priors",
                "2.13": "C · Errors and data weighting", "2.15": "D · Computing a posterior: sampling",
                "2.23": "E · Time-dependent and ensemble inversion", "2.27": "F · Hyperparameters and weights",
                "2.32": "G · Checking and synthesis"}
    items = []
    for key, stem, title in rows:
        if key == "3.01":
            items.append('</ul><h2>Topic 3 · Forward surrogate models <span class="tag">under development</span></h2><ul>')
        if key in sections:
            items.append(f'</ul><h3>{H.escape(sections[key])}</h3><ul>')
        t = H.escape(re.sub(r"^[23]\.\d\d · ", "", title))
        t = re.sub(r"\$\\mathbf ([A-Za-z])_([A-Za-z0-9]+)\$", r"<i>\1</i><sub>\2</sub>", t)   # $\mathbf C_d$ -> C<sub>d</sub>
        t = re.sub(r"\$([^$]+)\$", r"\1", t)                                               # any other inline math: plain
        items.append(f'<li><a href="{stem}.html"><b>{key}</b> {t}</a></li>')
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Empiricist Geophysics</title>
<style>
:root {{ --bg: #fcfcfb; --ink: #1d1d1b; --ink2: #52514e; --muted: #8a8984; --line: #e3e2dc; --accent: #2a78d6; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg: #141413; --ink: #ecebe6; --ink2: #b9b8b2; --muted: #8a8984;
  --line: #2c2c2a; --accent: #6aa6ee; }} }}
body {{ background: var(--bg); color: var(--ink); font: 16px/1.55 -apple-system, "Segoe UI", Arial, sans-serif;
  max-width: 860px; margin: 0 auto; padding: 24px 16px 64px; }}
h1 {{ margin-bottom: 4px; }} .sub {{ color: var(--ink2); margin-top: 0; }}
h2 {{ margin-top: 36px; border-bottom: 1px solid var(--line); padding-bottom: 4px; }}
h3 {{ color: var(--ink2); font-size: 1rem; margin: 22px 0 6px; }}
ul {{ list-style: none; padding: 0; margin: 0; }} li {{ padding: 3px 0; }}
a {{ color: var(--accent); text-decoration: none; }} a:hover {{ text-decoration: underline; }}
b {{ display: inline-block; min-width: 3.2em; color: var(--ink); }}
.tag {{ font-size: .7em; color: var(--muted); font-weight: normal; border: 1px solid var(--line); border-radius: 4px;
  padding: 1px 6px; vertical-align: middle; }}
.planned {{ color: var(--ink2); list-style: disc; padding-left: 1.2em; }} .planned li {{ padding: 2px 0; }}
.k {{ display: inline; min-width: 0; }} img {{ max-width: 100%; height: auto; }}
</style></head><body>
<h1>Empiricist Geophysics</h1>
<p class="sub"><b>Data, Models, and Bayesian Inference.</b> Geophysical inverse problems, by picture and by computation.
The notebooks I keep to refresh my own thinking and to build my own map of how data, models and inference fit together
in geophysics, shared as an open course. Every idea on a small problem whose answer is known; every method actually
run; every figure from that run.</p>
<p><a href="../figures/roadmap.png"><img src="../figures/roadmap.png" alt="Reading map of the course"></a></p>
<h2>Topic 1 · Geophysical forward problems and numerical methods <span class="tag">planned</span></h2>
<ul class="planned"><li><b class="k">Elastic dislocation models</b>: point sources and finite faults in an elastic half-space (Okada 1985, 1992), triangular dislocations and layered Earth models; how the Green's-function matrix G of a slip inversion is built, and how its errors become the prediction covariance C<sub>p</sub> (2.13).</li><li><b class="k">Numerical methods</b>: finite-difference, finite-element and boundary-element methods; meshing a curved fault from a slab model; checking accuracy by mesh refinement.</li><li><b class="k">Earthquake cycle and postseismic processes</b>: interseismic locking (backslip), rate-and-state and rate-strengthening friction, stress-driven afterslip (the forward model used in Topic 3), viscoelastic relaxation in Maxwell and Burgers materials, power-law flow laws, and poroelastic rebound.</li><li><b class="k">Tsunami modelling</b>: from seafloor deformation to the initial sea surface (including the contribution of horizontal motion on sloping seafloor), the shallow-water equations, dispersion (Boussinesq equations), adaptive mesh refinement, and tsunami Green's functions.</li><li><b class="k">Seismic waves</b>: the elastic wave equation, body and surface waves, synthetic seismograms, kinematic rupture models, and back projection of array recordings.</li></ul>
<h2>Topic 2 · Inverse problems and Bayesian inversion</h2><ul>
{chr(10).join(items)}
</ul>
<h2>Topic 4 · Data and data processing <span class="tag">planned</span></h2>
<ul class="planned"><li><b class="k">GNSS</b>: daily positions and velocities, high-rate GNSS displacement waveforms, reference frames, common-mode error, offsets, coloured noise (white, flicker and random walk; 2.14), and time-series models of postseismic motion.</li><li><b class="k">InSAR</b>: interferograms, phase unwrapping, atmospheric and orbital errors, decimation, and time-series methods; Sentinel-1, ALOS-2 and NISAR.</li><li><b class="k">Seafloor and ocean observations</b>: DART buoys; cabled seafloor networks such as S-net along the Japan Trench; ocean-bottom pressure and differential pressure gauges (DPG); GNSS-acoustic seafloor positioning; coastal tide gauges; and removing tides, instrument drift and non-tectonic signals.</li><li><b class="k">Satellite altimetry</b>: SWOT wide-swath sea-surface height and conventional nadir altimeters, as tsunami observations.</li><li><b class="k">Seismic data</b>: teleseismic body and surface waves, strong-motion accelerograms, instrument response, filtering and alignment.</li><li><b class="k">From data to C<sub>d</sub></b>: noise models, correlated errors and data weighting for each data type (2.14, 2.30).</li></ul>
<p class="sub" style="margin-top:28px">Also: <a href="../playground.html">the sampler playground</a> (runs in the browser).
Code MIT, text and figures CC BY 4.0.</p>
</body></html>
"""
    (HTML / "index.html").write_text(page.replace("<ul>\n</ul><h3>", "<h3>", 1))


if __name__ == "__main__":
    main()
