import sys
import pycparser
import difflib
import io

def rename_identifiers(node, name_map=None, counter=None):
    if name_map is None:
        name_map = {}
    if counter is None:
        counter = [0]  # Fresh counter for each function

    # Handle different node types that contain identifiers
    if isinstance(node, pycparser.c_ast.ID):
        if node.name not in name_map:
            name_map[node.name] = f'var{counter[0]}'
            counter[0] += 1
        node.name = name_map[node.name]

    # Handle function declarations
    elif isinstance(node, pycparser.c_ast.Decl) and isinstance(node.type, pycparser.c_ast.FuncDecl):
        if node.name not in name_map:
            name_map[node.name] = f'func{counter[0]}'
            counter[0] += 1
        node.name = name_map[node.name]
        # Also update the TypeDecl declname for function declaration
        if hasattr(node.type, 'type') and isinstance(node.type.type, pycparser.c_ast.TypeDecl):
            node.type.type.declname = node.name

    # Handle variable declarations
    elif isinstance(node, pycparser.c_ast.Decl) and isinstance(node.type, pycparser.c_ast.TypeDecl):
        if node.name not in name_map:
            name_map[node.name] = f'var{counter[0]}'
            counter[0] += 1
        node.name = name_map[node.name]
        # Also update the TypeDecl name to match
        node.type.declname = node.name

    # Handle TypeDecl nodes directly
    elif isinstance(node, pycparser.c_ast.TypeDecl):
        if node.declname is not None and node.declname in name_map:
            node.declname = name_map[node.declname]

    # Recursively process all children
    for _, child in node.children():
        rename_identifiers(child, name_map, counter)

def extract_function_ast(code, func_name):
    """Extract a specific function by name from the code."""
    parser = pycparser.CParser()
    ast = parser.parse(code)

    # Find the function in the AST
    for node in ast.ext:
        if (isinstance(node, pycparser.c_ast.FuncDef) and node.decl.name == func_name) or \
           (isinstance(node, pycparser.c_ast.Decl) and node.name == func_name and
            isinstance(node.type, pycparser.c_ast.FuncDecl)):
            # Create a new FileAST with just this function
            func_ast = pycparser.c_ast.FileAST([node])
            return func_ast

    raise ValueError(f"Function '{func_name}' not found in the code")

def normalize_ast(func_ast):
    """Normalize the AST of a function."""
    rename_identifiers(func_ast)
    buf = io.StringIO()
    func_ast.show(buf=buf)
    return buf.getvalue()

def compare_functions(func_name1, code1, func_name2, code2):
    """Compare two functions by name."""
    func_ast1 = extract_function_ast(code1, func_name1)
    func_ast2 = extract_function_ast(code2, func_name2)

    norm1 = normalize_ast(func_ast1).splitlines()
    norm2 = normalize_ast(func_ast2).splitlines()

    diff = difflib.unified_diff(norm1, norm2, lineterm='')
    return '\n'.join(diff)

# Example C functions
c_function1 = """
static inline _Bool mdns_utils_str_null_or_empty(const char *str)
{
    return (str == NULL || *str == 0);
}
"""

c_function2 = """
typedef _Bool bool;
typedef const char *string;

static inline _Bool _str_null_or_empty(const char *str)
{
    return (str == NULL || *str == 0);
}
"""

if __name__ == "__main__":
    result = compare_functions('mdns_utils_str_null_or_empty', c_function1, '_str_null_or_empty', c_function2)
    print("AST Differences:")
    print(result if result else "No significant differences.")
