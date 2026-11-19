import unittest

from seqsm.streaming import StreamingSeqSM


class TestStreamingSeqSM(unittest.TestCase):
    def test_online_updates_and_prediction(self):
        stream = StreamingSeqSM(
            k=1,
            mode="clock",
            decay=0.95,
            bins={"cpu": [0, 30, 70, 100]},
        )
        sequence = [10, 50, 80, 10, 50, 80, 10, 50]
        last = {}
        for val in sequence:
            last = stream.update({"cpu": val})

        self.assertEqual(last["event"], len(sequence))
        self.assertEqual(last["current"]["cpu"], "30-70")
        self.assertGreater(len(last["predictions"]), 0)
        self.assertEqual(last["predictions"][0]["state"]["cpu"], "70-100")
        self.assertGreater(len(last["graph"]["nodes"]), 0)
        self.assertGreater(len(last["graph"]["edges"]), 0)


if __name__ == "__main__":
    unittest.main()
