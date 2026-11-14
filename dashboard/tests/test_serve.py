from pathlib import Path
from http.server import ThreadingHTTPServer
from threading import Thread
import unittest
from unittest.mock import patch
from urllib.request import urlopen

from dashboard.serve import handler_for, live_html


class LiveDashboardTests(unittest.TestCase):
    def test_refresh_reads_run_each_time(self):
        with patch("dashboard.serve.load_run", return_value={"run": {"id": "run-test"}, "tasks": [], "library": {}}) as load:
            first = live_html(Path("/tmp/example"))
            second = live_html(Path("/tmp/example"))
        self.assertEqual(load.call_count, 2)
        self.assertIn('http-equiv="refresh" content="1"', first)
        self.assertIn('http-equiv="refresh" content="1"', second)
        self.assertIn("research-agent-task:", first)

    def test_split_screen_serves_two_distinct_runs(self):
        left, right = Path("/tmp/left-run"), Path("/tmp/right-run")
        with patch("dashboard.serve.live_html", side_effect=lambda path, **kwargs: str(path)):
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(left, Path("/tmp"), right))
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base = f"http://127.0.0.1:{server.server_port}"
                with urlopen(base + "/") as response:
                    self.assertIn('src="/right"', response.read().decode())
                with urlopen(base + "/left") as response:
                    self.assertEqual(response.read().decode(), str(left))
                with urlopen(base + "/right") as response:
                    self.assertEqual(response.read().decode(), str(right))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
