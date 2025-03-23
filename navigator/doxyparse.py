import xml.etree.ElementTree as ET
import os

def parse_doxygen_functions(xml_dir):
    functions = {}

    # Load the XML index
    index_path = os.path.join(xml_dir, "index.xml")
    print(f"Loading XML index from: {index_path}")  # Debug print
    tree = ET.parse(index_path)
    root = tree.getroot()

    # Extract function details from each XML file
    for compound in root.findall(".//compound"):
        print(f"Found compound: {compound.get('refid')} of kind: {compound.get('kind')}")  # Debug print
        if compound.get("kind") in ["file", "namespace"]:  # Limit to C/C++ files
            refid = compound.get("refid") + ".xml"
            compound_path = os.path.join(xml_dir, refid)
            
            if not os.path.exists(compound_path):
                print(f"Warning: XML file {compound_path} not found!")  # Debug print
                continue
            
            compound_tree = ET.parse(compound_path)
            compound_root = compound_tree.getroot()

            for member in compound_root.findall(".//memberdef[@kind='function']"):
                function_name = member.find("name").text
                
                location = member.find("location")
                file_path = location.get("file") if location is not None else "Unknown"
                start_line = int(location.get("line")) if location is not None and location.get("line") else None

                # Extract prototype
                definition = member.find("definition")
                prototype = definition.text if definition is not None else "Unknown"

                # Extract Doxygen comments (if available)
                doc = member.find("briefdescription")
                doxy_comment = "".join(doc.itertext()).strip() if doc is not None else ""

                # Store function details
                functions[function_name] = {
                    "file": file_path,
                    "line": start_line,
                    "prototype": prototype,
                    "doc": doxy_comment,
                }
                print(f"Added function: {function_name}, File: {file_path}, Line: {start_line}")  # Debug print

    print(f"Total functions parsed: {len(functions)}")  # Debug print
    return functions

# Example usage
xml_output_dir = "./xml"  # Adjust this if your doxygen XML is in another folder
function_data = parse_doxygen_functions(xml_output_dir)

for func, details in function_data.items():
    print(f"Function: {func}")
    print(f"File: {details['file']}, Line: {details['line']}")
    print(f"Prototype: {details['prototype']}")
    if details["doc"]:
        print(f"Documentation:\n{details['doc']}")
    print("-" * 50)
