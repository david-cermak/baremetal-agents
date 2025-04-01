import subprocess
import re

def get_function_ranges(filepath):
    # Run ctags to extract function names & line numbers
    ctags_output = subprocess.run(
        ["ctags", "--sort=no", "--fields=+n", "-x", filepath],
        capture_output=True, text=True
    ).stdout

    # Parse function start lines
    functions = {}
    for line in ctags_output.splitlines():
        match = re.match(r"(\S+)\s+function\s+(\d+)", line)
        if match:
            func_name, func_start = match.groups()
            functions[int(func_start)] = func_name

    # Find function end by tracking { and } balance
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    results = {}
    for start_line in sorted(functions.keys()):
        func_name = functions[start_line]
        brace_count = 0
        end_line = start_line

        for i in range(start_line - 1, len(lines)):
            line = lines[i]

            brace_count += line.count("{")
            brace_count -= line.count("}")

            if brace_count == 0 and i > start_line:
                end_line = i + 1
                break

        # Extract function content
        func_content = "".join(lines[start_line-1:end_line])
        results[func_name] = (start_line, end_line, func_content)

    return results

# # Example usage
# file_path = "/home/david/repos/proto/components/mdns/mdns_pcb.c"
# function_ranges = get_function_ranges(file_path)
# for func, (start, end, content) in function_ranges.items():
#     print(f"Function {func} starts at line {start} and ends at line {end}")
#     print("Content:")
#     print(content)
#     print("-" * 80)
