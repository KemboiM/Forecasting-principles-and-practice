"""
rmd_to_ipynb.py -- build a Python-only Jupyter notebook from a chapter .Rmd.

Every chapter .Rmd shows each step twice (an R tab and a Python tab). This script keeps
the explanations and the Python code, drops the R code, and writes <name>_python.ipynb next
to the .Rmd, so the Python half can also be run in Jupyter / VS Code / PyCharm.

Usage (from the project root):
    python python/rmd_to_ipynb.py                      # all chapters
    python python/rmd_to_ipynb.py Chapter-01-Getting-Started/ch01_notes.Rmd
"""
import re
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
CHUNK = re.compile(r"^```\{(\w+)([^}]*)\}\s*$")


def convert(rmd: Path) -> Path:
    text = rmd.read_text(encoding="utf-8")
    title = re.search(r'^title:\s*"(.*)"', text, re.M)
    subtitle = re.search(r'^subtitle:\s*"(.*)"', text, re.M)
    text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.S)  # drop YAML

    cells = []
    md_buf: list[str] = []
    header = f"# {title.group(1) if title else rmd.stem} (Python version)\n"
    if subtitle:
        header += f"\n*{subtitle.group(1)}*\n"
    header += ("\nAuto-generated from `" + rmd.name + "` by `python/rmd_to_ipynb.py` — "
               "the R code lives in the .Rmd file. Run this notebook from its own folder.")
    cells.append(nbf.v4.new_markdown_cell(header))

    def flush_md():
        s = "\n".join(md_buf).strip()
        s = re.sub(r"\n{3,}", "\n\n", s)
        if s:
            cells.append(nbf.v4.new_markdown_cell(s))
        md_buf.clear()

    lines = text.splitlines()
    i = 0
    skip_r_section = False
    while i < len(lines):
        line = lines[i]
        m = CHUNK.match(line)
        if m:
            engine = m.group(1).lower()
            j = i + 1
            body = []
            while j < len(lines) and not lines[j].startswith("```"):
                body.append(lines[j])
                j += 1
            if engine == "python":
                flush_md()
                code = "\n".join(body).strip("\n")
                if "setup-py" in m.group(2):
                    code = code.replace('os.path.abspath("../python")', 'os.path.abspath("../python")')
                    code += "\n%matplotlib inline"
                cells.append(nbf.v4.new_code_cell(code))
            i = j + 1
            continue
        # tab headings: "#### R" (skip until next heading) / "#### Python" (drop heading)
        if re.match(r"^#{2,6}\s+R\s*$", line):
            skip_r_section = True
            i += 1
            continue
        if re.match(r"^#{2,6}\s+Python\s*$", line):
            skip_r_section = False
            i += 1
            continue
        if re.match(r"^#{1,6}\s", line):
            skip_r_section = False
        if not skip_r_section:
            line = line.replace("{.tabset .tabset-pills}", "").replace("{.tabset}", "").rstrip()
            line = re.sub(r"`r [^`]+`", "", line)
            md_buf.append(line)
        i += 1
    flush_md()
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    out = rmd.with_name(rmd.stem + "_python.ipynb")
    nbf.write(nb, out)
    return out


if __name__ == "__main__":
    targets = [Path(a) for a in sys.argv[1:]] or sorted(ROOT.glob("Chapter-*/*.Rmd"))
    for t in targets:
        print("wrote", convert(t.resolve()).relative_to(ROOT))
