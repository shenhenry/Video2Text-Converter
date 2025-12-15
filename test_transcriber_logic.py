import unittest
from transcriber import time_to_ms, ms_to_time, shift_srt_content

class TestTranscriberLogic(unittest.TestCase):

    def test_time_to_ms(self):
        self.assertEqual(time_to_ms("00:00:01,000"), 1000)
        self.assertEqual(time_to_ms("00:01:00,000"), 60000)
        self.assertEqual(time_to_ms("01:00:00,000"), 3600000)
        self.assertEqual(time_to_ms("00:00:00,500"), 500)
        self.assertEqual(time_to_ms("01:30:15,123"), 3600000 + 1800000 + 15000 + 123)

    def test_ms_to_time(self):
        self.assertEqual(ms_to_time(1000), "00:00:01,000")
        self.assertEqual(ms_to_time(60000), "00:01:00,000")
        self.assertEqual(ms_to_time(3600000), "01:00:00,000")
        self.assertEqual(ms_to_time(500), "00:00:00,500")
        target_ms = 3600000 + 1800000 + 15000 + 123
        self.assertEqual(ms_to_time(target_ms), "01:30:15,123")

    def test_shift_srt_content(self):
        srt_input = """1
00:00:01,000 --> 00:00:02,000
Hello World

2
00:00:03,000 --> 00:00:04,500
Second Line"""

        offset_ms = 10000 # 10 seconds
        expected_srt = """1
00:00:11,000 --> 00:00:12,000
Hello World
2
00:00:13,000 --> 00:00:14,500
Second Line""" # Note: counter might reset or continue dependent on logic, let's check output

        shifted, last_counter = shift_srt_content(srt_input, offset_ms, counter_start=1)
        
        self.assertIn("00:00:11,000 --> 00:00:12,000", shifted)
        self.assertIn("00:00:13,000 --> 00:00:14,500", shifted)
        self.assertEqual(last_counter, 2)

    def test_shift_srt_content_continuation(self):
        srt_input = """1
00:00:01,000 --> 00:00:02,000
Test"""
        
        # Start counter at 10
        shifted, last_counter = shift_srt_content(srt_input, 0, counter_start=11)
        self.assertTrue(shifted.startswith("11"))
        self.assertEqual(last_counter, 11)

if __name__ == '__main__':
    unittest.main()
