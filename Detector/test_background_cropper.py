import unittest
from Detector.background_cropper import SubFrame


class TestBackgroundCropper(unittest.TestCase):
    def test_get_background_subframes_happy_flow(self):
        test1_input = [SubFrame(0, 0, 100, 100), {SubFrame(70, 10, 20, 10)}]
        test1_required_output = {SubFrame(0, 0, 100, 70),
                                 SubFrame(0, 0, 10, 100),
                                 SubFrame(80, 0, 100, 20),
                                 SubFrame(0, 30, 70, 100)}
        test1_results = SubFrame.get_background_subframes(*test1_input)
        self.assertEqual(test1_required_output, test1_results)

    def test_get_background_subframes_collision(self):
        test2_input = [SubFrame(0, 0, 100, 100), {SubFrame(65, 5, 30, 20), SubFrame(70, 10, 20, 10)}]
        test2_required_output = {SubFrame(0, 0, 100, 65),
                                 SubFrame(0, 0, 5, 100),
                                 SubFrame(0, 35, 65, 100),
                                 SubFrame(85, 0, 100, 15)}
        test2_results = SubFrame.get_background_subframes(*test2_input)
        self.assertEqual(test2_required_output, test2_results)

    def test_get_background_subframes_constrained(self):
        test3_input = [SubFrame(0, 0, 100, 100), {SubFrame(70, 10, 20, 10)}, 50, 50, 60 ** 2]
        test3_required_output = {SubFrame(0, 0, 100, 70), SubFrame(0, 30, 70, 100)}
        test3_results = SubFrame.get_background_subframes(*test3_input)
        self.assertEqual(test3_required_output, test3_results)

    def test_get_background_subframes_box_on_edge(self):
        test4_input = [SubFrame(0, 0, 100, 100), {SubFrame(0, 0, 50, 50)}]
        test4_required_output = {SubFrame(0, 50, 50, 100), SubFrame(50, 0, 100, 50)}
        test4_results = SubFrame.get_background_subframes(*test4_input)
        self.assertEqual(test4_required_output, test4_results)

    def test_get_background_subframes_box_consolidation(self):
        test5_input = [SubFrame(0, 0, 100, 100), {
            SubFrame(10, 10, 10, 10),
            SubFrame(10, 10, 10, 10),
            SubFrame(10, 10, 10, 10),
            SubFrame(10, 10, 10, 10)
        }]
        test5_required_output = {SubFrame(0, 0, 10, 100),
                                 SubFrame(0, 0, 100, 10),
                                 SubFrame(0, 20, 80, 100),
                                 SubFrame(20, 0, 100, 80)}
        test5_results = SubFrame.get_background_subframes(*test5_input)
        self.assertEqual(test5_required_output, test5_results)


if __name__ == "__main__":
    unittest.main()
