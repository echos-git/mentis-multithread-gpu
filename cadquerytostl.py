import cadquery as cq
from cadquery import exporters
import os
import argparse

PREFERRED_NAMES = ("result", "part", "model", "r") 

def _execute_cq(code: str) -> cq.Shape:
    """Execute a CadQuery script and return the first Workplane/Shape it creates."""
    ns: dict = {}
    exec(code, ns)
    for name in PREFERRED_NAMES:
        obj = ns.get(name)
        if isinstance(obj, cq.Workplane):
            return obj.val()
        if isinstance(obj, cq.Assembly):
            return obj.toCompound()
        if isinstance(obj, cq.Shape):
            return obj
    raise ValueError("No CadQuery Workplane or Shape found in the supplied code.")

def cq_to_stl(cq_script_path: str, output_path: str):
    """
    Convert a CadQuery script file to an STL file.
    
    Args:
        cq_script_path (str): Path to the CadQuery script.
        output_path (str): The path to save the STL file.
    """
    with open(cq_script_path, 'r') as f:
        cq_code = f.read()
        
    shape = _execute_cq(cq_code)
    exporters.export(shape, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CadQuery scripts to STL files.")
    parser.add_argument("input_path", help="Path to the CadQuery script.")
    parser.add_argument("output_path", help="Path to save the output STL file.")
    args = parser.parse_args()

    try:
        cq_to_stl(args.input_path, args.output_path)
        print(f"Successfully converted {args.input_path} to {args.output_path}")
    except Exception as e:
        print(f"Error converting {args.input_path}: {e}")
