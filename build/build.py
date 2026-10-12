"""Build the notebooks: build/src/NN_name.py (percent format) -> notebooks/NN_name.ipynb (executed) -> html/NN_name.html

usage:  python build/build.py              # all notebooks (2-01 ... 2-33, 3-01 ... 3-07)
        python build/build.py 2-07 2-08    # only those whose file name starts with 2-07 or 2-08
        python build/build.py 3-           # all of topic 3

Cell markers in the source:  "# %%" starts a code cell,  "# %% [markdown]" starts a markdown cell
(markdown lines are written as comments, "# text").
"""
from __future__ import annotations

import pathlib
import sys
import time

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC, NB, HTML = ROOT / "build" / "src", ROOT / "notebooks", ROOT / "html"


def py_to_nb(path: pathlib.Path) -> nbformat.NotebookNode:
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        while buf and not buf[-1].strip():
            buf.pop()
        while buf and not buf[0].strip():
            buf.pop(0)
        if not buf:
            return
        if kind == "markdown":
            text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in buf)
            cells.append(nbformat.v4.new_markdown_cell(text))
        else:
            cells.append(nbformat.v4.new_code_cell("\n".join(buf)))

    for line in path.read_text().splitlines():
        if line.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in line else "code"
            buf = []
        else:
            buf.append(line)
    flush()
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    return nb


def build(path: pathlib.Path):
    t0 = time.time()
    nb = py_to_nb(path)
    NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(NB)}}).execute()
    out = NB / (path.stem + ".ipynb")
    sys.path.insert(0, str(ROOT / "build"))
    import link_pages as L                                   # "Before this" / "Next" ids become links
    L.STEMS.update({path.stem[:4].replace("-", "."): path.stem})
    for c in nb.cells:
        if c.cell_type == "markdown":
            c.source = L.linkify(L.unlink(c.source), ".ipynb")
    nbformat.write(nb, out)
    for c in nb.cells:
        if c.cell_type == "markdown":
            c.source = L.linkify(L.unlink(c.source), ".html")
    body, _ = HTMLExporter(template_name="lab").from_notebook_node(nb)
    (HTML / (path.stem + ".html")).write_text(body)
    print(f"built {out.name} and html/{path.stem}.html in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    want = sys.argv[1:]
    import re
    srcs = sorted(q for q in SRC.glob("*.py") if re.match(r"^\d-\d\d_", q.name))   # <topic>-<number>_<name>.py
    for p in srcs:
        if not want or any(p.name.startswith(w) for w in want):
            build(p)
