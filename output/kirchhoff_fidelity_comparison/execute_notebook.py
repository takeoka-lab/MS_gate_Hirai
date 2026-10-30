"""Execute the delivered notebook using the existing project kernel."""
from pathlib import Path
import os
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[2]
output = root/'results/kirchhoff_fidelity_channel_comparison'
os.environ.setdefault('IPYTHONDIR', str(output/'.ipython'))
os.environ.setdefault('MPLCONFIGDIR', str(output/'.matplotlib'))
path = root/'kirchhoff_fidelity_channel_comparison.ipynb'
notebook = nbformat.read(path, as_version=4)


def progress(cell, cell_index, **kwargs):
    if cell.cell_type == 'code':
        print(f'Executing cell {cell_index+1}/{len(notebook.cells)}', flush=True)


client = NotebookClient(notebook, timeout=1800, kernel_name='ms-pauli-distance',
                        resources={'metadata': {'path': str(root)}},
                        on_cell_start=progress)
try:
    client.execute()
finally:
    nbformat.write(notebook, path)
errors = [o for c in notebook.cells if c.cell_type=='code'
          for o in c.outputs if o.output_type=='error']
print(f'Executed {len(notebook.cells)} cells; errors: {len(errors)}', flush=True)
