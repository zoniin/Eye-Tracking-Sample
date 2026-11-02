import unittest
from temporal_eyes import TemporalEyes, PositionSmoother, canonical_gaze_angles

class TemporalTests(unittest.TestCase):
    def test_blink_at_different_frame_rates(self):
        for fps in (15, 30, 60):
            tracker = TemporalEyes()
            tracker.update(0, .3)
            events = []
            for i in range(1, fps + 1):
                t = i / fps
                events.append(tracker.update(t, .1 if .2 <= t < .4 else .3))
            self.assertEqual(sum(e.blink for e in events), 1)

    def test_prolonged_closure_is_not_blink(self):
        tracker = TemporalEyes()
        for i in range(20):
            result = tracker.update(i / 10, .1)
        self.assertAlmostEqual(result.closure_seconds, 1.9)
        self.assertFalse(tracker.update(2, .3).blink)

    def test_missing_face_does_not_count_as_closure(self):
        tracker = TemporalEyes()
        tracker.update(0, .1)
        tracker.update(.1, None)
        result = tracker.update(.2, .3)
        self.assertFalse(result.blink)
        self.assertIsNone(result.closed_time_ratio)

    def test_long_gap_does_not_create_blink(self):
        tracker = TemporalEyes()
        tracker.update(0, .1)
        result = tracker.update(2, .3)
        self.assertFalse(result.blink)
        self.assertEqual(result.observed_seconds, 0)

    def test_clipped_time_window(self):
        tracker = TemporalEyes(window=.15)
        tracker.update(0, .1)
        tracker.update(.1, .3)
        result = tracker.update(.2, .3)
        self.assertAlmostEqual(result.observed_seconds, .15)
        self.assertAlmostEqual(result.closed_time_ratio, 1/3)

    def test_invalid_and_reversed_time(self):
        tracker = TemporalEyes()
        tracker.update(1, .1)
        result = tracker.update(0, float('nan'))
        self.assertIsNone(result.ear)
        self.assertEqual(result.observed_seconds, 0)
        with self.assertRaises(ValueError):
            tracker.update(float('nan'), .3)

    def test_smoothing_and_loss(self):
        smooth = PositionSmoother(2)
        smooth.update((0, 0), None)
        self.assertEqual(smooth.update((1, 1), None)[0], (.5, .5))
        smooth.update(None, None)
        self.assertEqual(smooth.update((1, 1), None)[0], (1, 1))

    def test_dataset_angle_order(self):
        self.assertEqual(canonical_gaze_angles([.2,.3], 'mpiigaze'),
                         canonical_gaze_angles([.3,.2], 'eve'))
        with self.assertRaises(ValueError):
            canonical_gaze_angles([float('nan'), 0], 'eve')

if __name__ == '__main__':
    unittest.main()
