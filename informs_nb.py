"""Load the notebook's own functions into one namespace.

The notebook stays the single source of truth: nothing here redefines a model
function.  Only cells whose top level is pure definition (imports, defs,
constants) are executed, so the cells that *run* long sweeps are skipped and
importing this module stays cheap and side-effect free.
"""
import ast
import json
import os

NOTEBOOK = os.environ.get(
    "INFORMS_NOTEBOOK",
    "/root/.claude/uploads/9656fb32-9251-5fdb-a2ca-46ddcbcf93ad/"
    "1116c0fb-Maria_iSHIPMENT_Finall.ipynb")

_PURE = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef,
         ast.ClassDef, ast.Assign, ast.AnnAssign, ast.If, ast.Try)

# Top-level calls that only configure the session or echo a value.  Allowing
# exactly these lets the figures cell through (it sets rcParams) without
# letting a cell that regenerates data or draws a chart run on import.
_BENIGN_CALLS = {
    "plt.rcParams.update", "matplotlib.rcParams.update", "mpl.rcParams.update",
    "plt.style.use", "matplotlib.use", "warnings.filterwarnings",
    "pd.set_option", "np.set_printoptions",
}


# A constant-folding assignment may call a container or maths constructor
# (BASE = dict(...), RANGES = dict(...)).  Anything else on the right-hand
# side means the cell builds an instance or runs a sweep, so it is skipped.
_SAFE_CALLS = {"dict", "list", "tuple", "set", "frozenset", "range", "int",
               "float", "str", "bool", "len", "sorted", "zip", "enumerate",
               "round", "abs", "min", "max", "sum"}
_SAFE_ROOTS = {"np", "numpy", "math", "itertools", "os", "re"}


def _dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _is_pure_assign(node):
    for sub in ast.walk(node.value):
        if not isinstance(sub, ast.Call):
            continue
        name = _dotted(sub.func)
        if name is None:
            return False
        if name in _SAFE_CALLS or name.split(".")[0] in _SAFE_ROOTS:
            continue
        return False
    return True


def _is_benign_expr(node):
    """A bare display line, or a call that only configures the session."""
    v = node.value
    if isinstance(v, (ast.Constant, ast.Name, ast.Subscript, ast.Attribute)):
        return True
    if isinstance(v, ast.Call):
        return _dotted(v.func) in _BENIGN_CALLS
    return False


def _is_definition_cell(src):
    """True when the cell only defines things -- no top-level calls."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    if not tree.body:
        return False
    for node in tree.body:
        if isinstance(node, ast.Expr) and _is_benign_expr(node):
            continue                      # docstring, display line or rcParams
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            if not _is_pure_assign(node):
                return False
            continue
        if not isinstance(node, _PURE):
            return False
    return True


def load(path=NOTEBOOK, verbose=False):
    """Return {name: object} with the notebook's functions and constants."""
    nb = json.load(open(path, encoding="utf-8"))
    ns = {"__name__": "informs_nb_cells"}
    used = []
    for i, cell in enumerate(c for c in nb["cells"] if c["cell_type"] == "code"):
        src = "".join(cell["source"])
        if not _is_definition_cell(src):
            continue
        exec(compile(src, f"<notebook cell {i}>", "exec"), ns)
        used.append(i)
    ns["_cells_executed"] = used
    if verbose:
        print(f"notebook {os.path.basename(path)}: executed cells {used}")
    return ns


REQUIRED = ["build_model", "solve", "extract", "itinerary", "pareto_front",
            "knee_point", "fig_swimlane", "stages", "build_instance",
            "BASE", "GRID", "front_for", "eligibility_coefficients"]


def load_checked(path=NOTEBOOK, verbose=True):
    ns = load(path, verbose=verbose)
    missing = [n for n in REQUIRED if n not in ns]
    if missing:
        raise RuntimeError(f"notebook is missing {missing}")
    return ns
