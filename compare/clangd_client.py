import json
import subprocess
import os
from threading import Thread, Event
import time
class ClangdClient:
    def __init__(self, project_root=None):
        self.project_root = project_root
        self.process = subprocess.Popen(
            ['clangd', '--background-index', '--log=verbose',
             '--compile-commands-dir=/home/david/repos/proto/components/mdns/tests/host_unit_test/build2',
            #  '--query-driver=/usr/bin/cc',
            #  '-j=1', '--pch-storage=memory'],
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            bufsize=0,
            universal_newlines=True
        )
        self.seq = 1
        self.initialized = Event()
        Thread(target=self.read_output).start()
        self.initialize()

    def initialize(self):
        """Initialize the LSP server before making any other requests."""
        init_params = {
            "processId": subprocess.Popen("echo $$", shell=True, stdout=subprocess.PIPE).stdout.read().decode().strip(),
            "rootUri": f"file://{self.project_root}" if self.project_root else None,
            "capabilities": {
                "textDocument": {
                    "definition": {"dynamicRegistration": True}
                }
            },
            "initializationOptions": {
                "compilationDatabasePath": "/home/david/repos/proto/components/mdns/tests/host_unit_test/build2"
            }
        }
        self.send_request("initialize", init_params)
        # Wait for server to be initialized
        self.initialized.wait(timeout=5)
        # Send initialized notification
        self.send_notification("initialized", {})

    def send_request(self, method, params):
        request = {
            "jsonrpc": "2.0",
            "id": self.seq,
            "method": method,
            "params": params
        }
        request_json = json.dumps(request)
        self.process.stdin.write(f"Content-Length: {len(request_json)}\r\n\r\n{request_json}\r\n")
        self.process.stdin.flush()
        self.seq += 1

    def send_notification(self, method, params):
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        }
        notification_json = json.dumps(notification)
        self.process.stdin.write(f"Content-Length: {len(notification_json)}\r\n\r\n{notification_json}\r\n")
        self.process.stdin.flush()

    def read_output(self):
        while True:
            content_length = None
            while True:
                line = self.process.stdout.readline().strip()
                if not line:
                    break
                if line.startswith('Content-Length:'):
                    content_length = int(line.split(': ')[1])

            if content_length:
                raw_data = self.process.stdout.read(content_length)
                response = json.loads(raw_data)
                self.handle_response(response)

    def handle_response(self, response):
        if 'result' in response:
            print("Received result:", response['result'])
            # Signal that initialization is complete if this was the initialize response
            if response.get('id') == 1:
                self.initialized.set()
        elif 'error' in response:
            print("Error:", response['error'])

    def get_definition(self, file_path, line, character):
        self.send_request("textDocument/definition", {
            "textDocument": {"uri": f"file://{file_path}"},
            "position": {"line": line, "character": character}
        })

    def find_references(self, file_path, line, character):
        self.send_request("textDocument/references", {
            "textDocument": {"uri": f"file://{file_path}"},
            "position": {"line": line, "character": character},
            "context": {"includeDeclaration": True}
        })

    def did_open(self, file_path):
        """Notify the server that a file is open and ready for language features."""
        with open(file_path, 'r') as f:
            content = f.read()

        self.send_notification("textDocument/didOpen", {
            "textDocument": {
                "uri": f"file://{file_path}",
                "languageId": "c",
                "version": 1,
                "text": content
            }
        })

# Example usage
if __name__ == "__main__":
    # Change to the project directory first
    # os.chdir("/home/david/repos/proto/components/mdns/tests/host_unit_test/build2")
    # os.chdir("/home/david/repos/proto/")

    # Create client with project root
    # client = ClangdClient(os.getcwd())
    client = ClangdClient()

    # Now reference files relative to the project root
    file_path = "/home/david/repos/proto/components/mdns/mdns_querier.c"
    # abs_file_path = os.path.join(os.getcwd(), file_path)

    # Open the document
    client.did_open(file_path)
    # client.did_open("/home/david/repos/proto/components/mdns/private_include/mdns_querier.h")
    # time.sleep(1)
    # Then request definitions
    client.get_definition(file_path, 7, 5) # Line 5 (0-indexed)
