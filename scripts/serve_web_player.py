"""Siren-Zip: WebGPU In-Browser Cinema Player Local HTTP Server.

Serves the WebGPU web application with correct MIME types for WGSL compute shaders,
Cross-Origin Isolation headers, and auto-launches the default web browser.
"""

from __future__ import annotations

import argparse
import http.server
import mimetypes
import os
import socketserver
import sys
import threading
import time
import webbrowser

# Ensure UTF-8 output on Windows
if sys.stdout.encoding != "utf-8" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure correct MIME type mappings
mimetypes.add_type("text/plain", ".wgsl")
mimetypes.add_type("application/wgsl", ".wgsl")
mimetypes.add_type("application/octet-stream", ".neura")
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("application/json", ".json")


class WebGPUHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP Request Handler adding Cross-Origin Isolation and WGSL MIME types."""

    def __init__(self, *args, directory=None, **kwargs):
        if directory is None:
            # Serve from project root so both web/ and .neura files are accessible
            directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        super().__init__(*args, directory=directory, **kwargs)

    def do_GET(self) -> None:
        # Redirect root '/' to '/web/index.html'
        if self.path == "/" or self.path == "":
            self.send_response(302)
            self.send_header("Location", "/web/index.html")
            self.end_headers()
            return
        super().do_GET()

    def end_headers(self) -> None:
        # Add Cross-Origin Isolation headers for WebGPU / SharedArrayBuffer high performance
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache, must-revalidate")
        super().end_headers()

    def guess_type(self, path: str) -> str:
        if path.endswith(".wgsl"):
            return "text/plain; charset=utf-8"
        if path.endswith(".neura"):
            return "application/octet-stream"
        if path.endswith(".js"):
            return "application/javascript; charset=utf-8"
        return super().guess_type(path)


def serve(port: int = 8000, open_browser: bool = True) -> None:
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    web_dir = os.path.join(root_dir, "web")

    if not os.path.exists(web_dir):
        print(f"Error: Web directory not found at {web_dir}", flush=True)
        sys.exit(1)

    handler = lambda *args, **kwargs: WebGPUHTTPRequestHandler(*args, directory=root_dir, **kwargs)

    # Allow port reuse
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", port), handler) as httpd:
        url = f"http://localhost:{port}/web/index.html"
        print("\n==========================================================================")
        print("🌐 SIREN-ZIP: WEBGPU ZERO-INSTALL IN-BROWSER PLAYER")
        print("==========================================================================")
        print(f"   Root Directory     : {root_dir}")
        print(f"   Web App URL        : {url}")
        print(f"   Supported Browsers : Google Chrome 113+, Microsoft Edge, Safari 18+")
        print(f"   Features           : 60 FPS WGSL Shaders, 400X Zoom, Drag & Drop, Live Stream")
        print("==========================================================================\n")
        print(f"🟢 Server listening on port {port}. Press Ctrl+C to stop.\n", flush=True)

        if open_browser:
            def _open():
                time.sleep(0.5)
                webbrowser.open(url)
            threading.Thread(target=_open, daemon=True).start()

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[SERVER] Web server stopped.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Siren-Zip WebGPU Browser Player.")
    parser.add_argument("--port", type=int, default=8000, help="Local HTTP port")
    parser.add_argument("--no_browser", action="store_true", help="Do not automatically launch web browser")
    args = parser.parse_args()

    serve(port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
