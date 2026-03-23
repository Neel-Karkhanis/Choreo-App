import sys
import types
import unittest
from unittest.mock import MagicMock, patch


moviepy_module = types.ModuleType("moviepy")
editor_module = types.ModuleType("moviepy.editor")
editor_module.VideoFileClip = MagicMock(name="VideoFileClip")
editor_module.vfx = types.SimpleNamespace(speedx=object(), mirror_x=object())
moviepy_module.editor = editor_module
sys.modules.setdefault("moviepy", moviepy_module)
sys.modules["moviepy.editor"] = editor_module

import src.video_controls as video_controls


class TestVideoControls(unittest.TestCase):
    @patch("video_controls.VideoFileClip")
    @patch("video_controls.os.path.exists", return_value=True)
    def test_load_video_returns_clip_for_existing_file(self, mock_exists, mock_video_file_clip):
        """load_video returns the created clip when the file exists."""
        expected_clip = MagicMock()
        mock_video_file_clip.return_value = expected_clip

        result = video_controls.load_video("sample.mp4")

        mock_exists.assert_called_once_with("sample.mp4")
        mock_video_file_clip.assert_called_once_with("sample.mp4")
        self.assertIs(result, expected_clip)

    @patch("video_controls.os.path.exists", return_value=False)
    def test_load_video_raises_for_missing_file(self, mock_exists):
        """load_video raises FileNotFoundError for a missing path."""
        with self.assertRaises(FileNotFoundError):
            video_controls.load_video("missing.mp4")

        mock_exists.assert_called_once_with("missing.mp4")

    def test_change_speed_applies_speed_effect(self):
        """change_speed applies MoviePy's speed effect for valid input."""
        clip = MagicMock()
        expected_result = MagicMock()
        clip.fx.return_value = expected_result

        result = video_controls.change_speed(clip, 1.5)

        clip.fx.assert_called_once_with(video_controls.vfx.speedx, 1.5)
        self.assertIs(result, expected_result)

    def test_change_speed_raises_for_non_positive_speed(self):
        """change_speed rejects zero or negative speed multipliers."""
        clip = MagicMock()

        with self.assertRaises(ValueError):
            video_controls.change_speed(clip, 0)

        clip.fx.assert_not_called()

    def test_loop_section_returns_subclip_for_valid_range(self):
        """loop_section returns a subclip when the range is valid."""
        clip = MagicMock()
        clip.duration = 12.0
        expected_section = MagicMock()
        clip.subclip.return_value = expected_section

        result = video_controls.loop_section(clip, 2.0, 5.0)

        clip.subclip.assert_called_once_with(2.0, 5.0)
        self.assertIs(result, expected_section)

    def test_loop_section_raises_when_points_are_out_of_bounds(self):
        """loop_section raises when start or end is outside the clip duration."""
        clip = MagicMock()
        clip.duration = 10.0

        with self.assertRaises(ValueError):
            video_controls.loop_section(clip, -1.0, 5.0)

        with self.assertRaises(ValueError):
            video_controls.loop_section(clip, 2.0, 11.0)

        clip.subclip.assert_not_called()

    def test_loop_section_raises_when_start_is_not_before_end(self):
        """loop_section raises when the start time is not before the end time."""
        clip = MagicMock()
        clip.duration = 10.0

        with self.assertRaises(ValueError):
            video_controls.loop_section(clip, 4.0, 4.0)

        with self.assertRaises(ValueError):
            video_controls.loop_section(clip, 6.0, 4.0)

        clip.subclip.assert_not_called()

    def test_get_frame_returns_frame_for_valid_timestamp(self):
        """get_frame returns the frame at a valid timestamp."""
        clip = MagicMock()
        clip.duration = 8.0
        expected_frame = MagicMock()
        clip.get_frame.return_value = expected_frame

        result = video_controls.get_frame(clip, 3.5)

        clip.get_frame.assert_called_once_with(3.5)
        self.assertIs(result, expected_frame)

    def test_get_frame_raises_for_invalid_timestamp(self):
        """get_frame raises when the timestamp falls outside the clip."""
        clip = MagicMock()
        clip.duration = 8.0

        with self.assertRaises(ValueError):
            video_controls.get_frame(clip, -0.1)

        with self.assertRaises(ValueError):
            video_controls.get_frame(clip, 8.1)

        clip.get_frame.assert_not_called()

    def test_step_frames_moves_forward_by_one_frame(self):
        """step_frames advances by one frame when moving forward."""
        clip = MagicMock()
        clip.fps = 25
        clip.duration = 10.0

        result = video_controls.step_frames(clip, 1.0, "forward")

        self.assertAlmostEqual(result, 1.04)

    def test_step_frames_caps_forward_movement_at_duration(self):
        """step_frames does not move past the clip duration."""
        clip = MagicMock()
        clip.fps = 30
        clip.duration = 5.0

        result = video_controls.step_frames(clip, 4.99, "forward")

        self.assertEqual(result, 5.0)

    def test_step_frames_moves_backward_by_one_frame(self):
        """step_frames rewinds by one frame when moving backward."""
        clip = MagicMock()
        clip.fps = 20
        clip.duration = 10.0

        result = video_controls.step_frames(clip, 1.0, "backward")

        self.assertAlmostEqual(result, 0.95)

    def test_step_frames_caps_backward_movement_at_zero(self):
        """step_frames does not move before the start of the clip."""
        clip = MagicMock()
        clip.fps = 24
        clip.duration = 10.0

        result = video_controls.step_frames(clip, 0.01, "backward")

        self.assertEqual(result, 0)

    def test_step_frames_raises_for_invalid_direction(self):
        """step_frames raises for directions other than forward or backward."""
        clip = MagicMock()
        clip.fps = 24
        clip.duration = 10.0

        with self.assertRaises(ValueError):
            video_controls.step_frames(clip, 1.0, "sideways")
    
    def test_mirror_video(self):
        '''mirror_video applies MoviePy's mirror_x effect when called'''
        clip = MagicMock()
        expected_result = MagicMock()
        clip.fx.return_value = expected_result

        result = video_controls.mirror_video(clip)

        clip.fx.assert_called_once_with(video_controls.vfx.mirror_x)
        self.assertIs(result, expected_result)


if __name__ == "__main__":
    unittest.main()
