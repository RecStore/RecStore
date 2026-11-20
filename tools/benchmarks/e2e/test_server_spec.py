import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.benchmarks.e2e.custom.config import parse_server_spec  # noqa: E402


class TestParseServerSpec(unittest.TestCase):
    def test_defaults(self):
        server = parse_server_spec("ssh_host=local")
        self.assertEqual(server.port, 15000)
        self.assertEqual(server.shard_id, 0)

    def test_ps_port_is_accepted(self):
        # The documented spelling in .agents/skills/rs-benchmark-e2e; a silent
        # fallback to 15000 makes every shard collide on the same port.
        server = parse_server_spec("ssh_host=local,ip=127.0.0.1,ps_port=15001,shard_id=1")
        self.assertEqual(server.port, 15001)
        self.assertEqual(server.shard_id, 1)

    def test_port_beats_ps_port(self):
        server = parse_server_spec("ssh_host=local,port=15002,ps_port=15001")
        self.assertEqual(server.port, 15002)

    def test_two_shards_keep_distinct_ports(self):
        first = parse_server_spec("ssh_host=local,ip=127.0.0.1,ps_port=15000,shard_id=0")
        second = parse_server_spec("ssh_host=local,ip=127.0.0.1,ps_port=15001,shard_id=1")
        self.assertNotEqual(first.port, second.port)


if __name__ == "__main__":
    unittest.main()
