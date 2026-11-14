"""Local, read-only live dashboard for a running or completed run."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .export import PROJECT_ROOT, load_run, render_html


def live_html(run_dir: Path, *, project_root: Path = PROJECT_ROOT) -> str:
    """Render fresh artifacts on each request; refresh every second."""
    html = render_html(load_run(run_dir, project_root=project_root))
    marker = "</head>"
    if marker not in html:
        raise ValueError("Dashboard template lacks a head section")
    return html.replace(marker, '<meta http-equiv="refresh" content="1">' + marker, 1)


def handler_for(run_dir: Path, project_root: Path, other_run_dir: Path | None = None):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = urlsplit(self.path).path
            if route not in ({"/", "/index.html", "/left", "/right"} if other_run_dir else {"/", "/index.html"}):
                self.send_error(404)
                return
            try:
                if other_run_dir and route in {"/", "/index.html"}:
                    html = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
                            '<title>Research Agent · 分屏演示</title>'
                            '<style>body{margin:0;font:14px system-ui;background:#edf1f8}'
                            'header{padding:12px 20px;background:#14213b;color:white}'
                            'main{display:grid;grid-template-columns:1fr 1fr;height:calc(100vh - 45px)}'
                            'iframe{border:0;width:100%;height:100%}</style>'
                            '<header>Research Agent · 左：空库 / 右：练过的库 · 演示结果不计入正式报告</header>'
                            '<main><iframe src="/left" title="空库运行"></iframe>'
                            '<iframe src="/right" title="练过的库运行"></iframe></main></html>')
                else:
                    selected = other_run_dir if route == "/right" else run_dir
                    html = live_html(selected, project_root=project_root)
                body = html.encode("utf-8")
            except (OSError, ValueError, KeyError) as exc:
                self.send_error(503, str(exc))
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve a read-only, one-second run dashboard")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--other-run-dir", type=Path, help="Show two runs side by side")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    if args.host not in {"127.0.0.1", "::1", "localhost"}:
        parser.error("Live dashboard is intentionally limited to localhost")
    with ThreadingHTTPServer((args.host, args.port), handler_for(args.run_dir, args.project_root, args.other_run_dir)) as server:
        print(f"Serving http://{args.host}:{server.server_port}/")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
