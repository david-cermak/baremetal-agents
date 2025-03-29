import xml.etree.ElementTree as ET
import os
import re
from io import StringIO

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
                brief_doc = member.find("briefdescription")
                detailed_doc = member.find("detaileddescription")

                # Get brief description
                brief_text = "".join(brief_doc.itertext()).strip() if brief_doc is not None else ""

                # Process detailed description section by section
                detailed_text = ""
                param_text = ""
                return_text = ""

                if detailed_doc is not None:
                    # Get parameters
                    param_list = detailed_doc.find(".//parameterlist[@kind='param']")
                    if param_list is not None:
                        param_text = "Parameters:\n"
                        for param_item in param_list.findall("parameteritem"):
                            name = param_item.find(".//parametername").text if param_item.find(".//parametername") is not None else "Unknown"
                            desc = "".join(param_item.find(".//parameterdescription").itertext()).strip() if param_item.find(".//parameterdescription") is not None else "No description"
                            param_text += f"- *{name}* {desc}\n"

                    # Get return value
                    return_sect = detailed_doc.find(".//simplesect[@kind='return']")
                    if return_sect is not None:
                        return_value = "".join(return_sect.itertext()).strip()
                        if return_value:
                            return_text = f"Returns:\n{return_value}"

                    # Extract detailed description text without parameters and return value
                    # We'll use a different approach - convert to string and manually clean up

                    # Convert the detailed_doc to string
                    detailed_xml = ET.tostring(detailed_doc, encoding='utf-8').decode('utf-8')

                    # Remove parameterlist sections
                    detailed_xml = re.sub(r'<parameterlist.*?</parameterlist>', '', detailed_xml, flags=re.DOTALL)

                    # Remove simplesect (return) sections
                    detailed_xml = re.sub(r'<simplesect.*?</simplesect>', '', detailed_xml, flags=re.DOTALL)

                    # Convert back to element
                    temp_tree = ET.parse(StringIO(f'<root>{detailed_xml}</root>'))
                    temp_root = temp_tree.getroot()

                    # Extract text
                    detailed_text = "".join(temp_root.itertext()).strip()

                    # Clean up extra whitespace
                    detailed_text = re.sub(r'\s+', ' ', detailed_text).strip()

                # Combine all sections
                doxy_comment = brief_text

                if detailed_text:
                    doxy_comment += "\n\n" + detailed_text

                if param_text:
                    doxy_comment += "\n\n" + param_text

                if return_text:
                    doxy_comment += "\n\n" + return_text

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
