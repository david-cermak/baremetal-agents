from openai import OpenAI
import mdns_parser
import os
import re
import subprocess

GCOV_DIR="/home/david/repos/proto/components/mdns/tests/test_afl_fuzz_host"
GCOV_FILE=GCOV_DIR+"/mdns_receive.c.gcov"
MAX_RUNS=10
for run_id in range(MAX_RUNS):
    print(f"Run {run_id}")
    i = 0
    while os.path.exists(f"{GCOV_DIR}/run_{i}"):
        i += 1
    run_dir = f"{GCOV_DIR}/run_{i}"
    os.makedirs(run_dir)

    with open(GCOV_FILE, "rb") as f:
        code_under_test = f.read()

    messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful assistant specializing in generating test cases to maximize code coverage. "
            "Your output should be a clear, well-commented Python script that uses the `create_mdns_packet()` API. "
            "You should include various test cases, including edge cases, to ensure that every part of the mDNS parsing "
            "code (provided by the user) is exercised. Feel free to use loops and manual entries, and provide brief comments "
            "for each test case explaining its purpose."
        )
    },
    {
        "role": "user",
        "content": 
        f"""
I have an mDNS packet parsing implementation. Below is a snippet of my mDNS code for parsing packets:
It is gcov output file showing which lines are covered.
```c
{code_under_test}
```

Additionally, here's a Python script used to generate mDNS packets:

```python
{mdns_parser.dns_generate}
```

I also have a test file that feeds the packet content to the mDNS parsing code and measures code coverage:

```c
// mdns-receive test code
{mdns_parser.test_code}
```
Could you please generate a Python script that creates test cases by calling the create_mdns_packet() API? 
Your script should aim to maximize code coverage of the mDNS parsing code by generating multiple valid (or invalid) and edge-case mDNS packets.
Focus on lines which were not covered in the code. Packet input function is `void mdns_parse_packet(mdns_rx_packet_t *packet)`.
Use the information from "mdns-receive test code" (about which services, names, and types are used) to generate the test cases.
Please output only directly executable python code (you can add comments and prints to the code).""" } ]

    # print(messages)

    # client = OpenAI(api_key=os.environ["API_KEY"], base_url="https://api.deepseek.com")
    client = OpenAI(api_key=os.environ["API_KEY"], base_url=os.environ["BASE_URL"])
    model = os.environ["MODEL"]

    completion = client.chat.completions.create(model=model, messages=messages )

    if not completion or not getattr(completion, "choices", None) or not completion.choices:
        print("Completion failed... continue")
        continue
    print("after completion")
    print(completion.choices[0].message.content)

    completion_text = completion.choices[0].message.content
    match = re.search(r'```python\n(.*?)```', completion_text, re.DOTALL)

    if match:
        extracted_code = match.group(1)
        script_path = os.path.join(run_dir, "generate_cases.py")
        
        with open(script_path, "w") as f:
            f.write(extracted_code)
        
        print(f"Python script saved to {script_path}")
        parent_dir = os.path.abspath(os.path.join(run_dir, ".."))
        
        # Step 3: Execute the generated Python script
        try:
            subprocess.run(["python", script_path], cwd=run_dir, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing generated script: {e}")
            print("...but let's continue, maybe we have some valid test cases.")
        # Step 4: Check for .bin files and execute test_sim
        bin_files = [f for f in os.listdir(run_dir) if f.endswith(".bin")]
        print("\nbin_files:")
        print(bin_files)
        
        if bin_files:
            for bin_file in bin_files:
                bin_path = os.path.join(run_dir, bin_file)
                print(f"\nRunning test_sim with {bin_path}")
                subprocess.run(["./test_sim", bin_path], cwd=parent_dir, check=True)
            
            # Step 5: Run gcov to update coverage info
            print("\nRunning gcov to update coverage info...")
            subprocess.run(["gcov", "mdns_receive.c"], cwd=parent_dir, check=True)    
        
    else:
        print("No valid Python code found in the completion output.")

