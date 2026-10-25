"""Build both manuscript entry points using the local multi-file TeX toolchain."""
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parents[1]
for name in ['main','main_preprint']:
 for command in [['pdflatex','-interaction=nonstopmode','-halt-on-error',name+'.tex'],['bibtex',name],['pdflatex','-interaction=nonstopmode','-halt-on-error',name+'.tex'],['pdflatex','-interaction=nonstopmode','-halt-on-error',name+'.tex']]:
  completed=subprocess.run(command,cwd=root/'manuscript',capture_output=True,text=True)
  if completed.returncode:
   print(completed.stdout[-5000:]); raise SystemExit(completed.returncode)
 log=(root/'manuscript'/f'{name}.log').read_text()
 for failure in ['undefined citations','undefined references','Overfull','! LaTeX Error']:
  if failure in log:raise RuntimeError(f'{name}: {failure}; inspect compiler log')
 print(f'Built manuscript/{name}.pdf')
