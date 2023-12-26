import unittest
import json


class TestOutputFormat(unittest.TestCase):

    def assertJsonStructure(self, data, expected_structure):
        # Recursively check the structure of nested dictionaries
        for key, value in expected_structure.items():
            self.assertIn(key, data)
            if isinstance(value, dict):
                self.assertDictStructure(data[key], value)
            else:
                self.assertIsNone(value)  # Ensure that the value is None

    def assertDictStructure(self, data, expected_structure):
        # Recursively check the structure of nested dictionaries
        for key, value in expected_structure.items():
            self.assertIn(key, data)
            if isinstance(value, dict):
                self.assertDictStructure(data[key], value)
            else:
                self.assertIsNone(value)  # Ensure that the value is None

    def checkJsonStructure(self, data, expected_structure, path=''):
        """
        Recursively checks if the JSON data matches the expected structure.
        """
        for key, value in expected_structure.items():
            current_path = f"{path}.{key}" if path else key

            assert key in data, f"Key '{current_path}' is missing in the JSON data."

            if isinstance(value, dict):
                assert isinstance(
                    data[key], dict), f"Expected '{current_path}' to be a dictionary."
                self.checkJsonStructure(data[key], value, current_path)
            elif isinstance(value, type):
                assert isinstance(
                    data[key], value), f"Expected '{current_path}' to be of type {value.__name__}."
            # elif isinstance(value,(list,tuple,bool,))
            else:
                continue
                raise ValueError(
                    "Invalid value type in the expected structure.")

        # Check for extra keys in the data
        extra_keys = set(data.keys()) - set(expected_structure.keys())
        assert not extra_keys, f"Unexpected keys found: {', '.join(extra_keys)}."

    def test_output_format(self):
        # Replace 'path/to/expected_structure.json' with the actual path to your JSON file
        expected_structure_file_path = '/home/akhilesh/research/sneaker-dev/tests/output_format/format_authentic_shape_box_tag_nike_lightbox.json'

        try:
            with open(expected_structure_file_path, 'r') as file:
                expected_structure = json.load(file)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            self.fail(f"Error loading expected structure: {str(e)}")

        module, entrypoint, config, request = "entrupy.ml_modules.sneakers.submit.production_sneaker_box", "sneaker_auth_ocr_v1", "/home/akhilesh/research/sneaker-dev/ray/entrupy/ml_modules/sneakers/submit/sneakers_box_label_ocr_auth_pose.json", "/home/akhilesh/research/sneaker-dev/ray/entrupy/ml_modules/sneakers/submit/request_artifacts/sample_jordan1_high_request_ocr.json"  # -l --warmup-iterations 1 --keep-alive &
        actual_output = mock_production_run(
            module, entrypoint, request, config)
        # # Replace this with your actual JSON output
        # actual_output = {
        #     # Replace with your JSON data
        # }
        # self.assertJsonStructure(actual_output[list(actual_output.keys())[0]], expected_structure[list(actual_output.keys())[0]])

        # Parse the JSON data
        try:
            actual_data = actual_output
        except json.JSONDecodeError:
            self.fail("Invalid JSON format")
        # breakpoint()
        # Check if the parsed data has the expected structure
        self.checkJsonStructure(actual_data, expected_structure)


def mock_production_run(module, entrypoint, request, config, cluster_logs=False, warmup_iterations=0, keep_alive=False, persist=False):
    """Run production entrypoint locally."""
    import os
    os.environ['ENTRUPY_RAY_RUN_MODE'] = 'production'
    # start throwing errors etc. if they do bad things for production
    os.environ['ENTRUPY_RAY_PRODUCTION_LINTING'] = 'true'
    import entrupy.ray as ray  # do not import at the top just in case
    import importlib
    import json
    importlib.import_module(module)
    import entrupy.ray._productionization._interface_v1 as ifv1
    from entrupy.ray._productionization._shared import format_output
    import entrupy.ray._productionization._models as models
    with open(request,'rb') as f:
        request = json.load(f)
    with open(config,'rb') as f:
        config = json.load(f)
    # assert request.keys() == {'artifacts', 'session_props', 'metadata_props', 'request_context'} # XXX: validate by template type
    request['config'] = config
    # breakpoint()
    print("Loaded successfully, running...")
    ray.init(log_to_driver=cluster_logs)

    for i in range(warmup_iterations):
        print(f"Warmup run {i+1}...")
        ifv1._REGISTERED_TEMPLATES[entrypoint].run_mock(request)
    if warmup_iterations:
        print("Warmed up.")

    # Run for real...
    with ray.Job(), (ray.PersistGroup() if persist else ray.DummyContextManager()):
        output_format = models.OutputFormat(
            response_type='blocking_get', output_source='output.json')
        output = ifv1._REGISTERED_TEMPLATES[entrypoint].run_mock(request)
        output_json = format_output(output, output_format)
        output_bytes = output_json
    return output_bytes
    #     print(
    #         f"output.json ({len(output_bytes)} bytes):",
    #         output_json,
    #     )
    #     if len(output_bytes) >= 33000:
    #         print("ERROR: Output must be under 32kb or else it will be too big for the database - talk to the ML platform team if you want bigger")

    # if keep_alive:
    #     breakpoint()


if __name__ == '__main__':

    unittest.main()
