"""
Tests for Data Processor Module.
"""

import unittest
import pandas as pd
import os
import json
from src.data_processor import OrphanetDataProcessor  # Correct import
from pandas.testing import assert_frame_equal

class TestOrphanetDataProcessor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a temporary directory and sample CSV file for testing
        cls.test_data_dir = "test_data"
        os.makedirs(cls.test_data_dir, exist_ok=True)
        cls.test_file_path = os.path.join(cls.test_data_dir, "test_data.csv")

        # Sample data for testing
        data = {
            'OrphaCode': [1, 2],
            'Name': ['Disease A', 'Disease B'],
            'HPODisorderAssociation_df2': [
                '[{"HPOId": "HP:0001", "HPOTerm": "Term1", "HPOFrequency": "Frequent (79-30%)"}]',
                '[{"HPOId": "HP:0002", "HPOTerm": "Term2", "HPOFrequency": "Very rare (<5%)"}]'
            ],
             'DisabilityDisorderAssociations_df3': [
                '[{"Disability": "Walking", "FrequencyDisability": "Frequent"}]',
                '[{"Disability": "Running", "FrequencyDisability": "Very rare (<5%)"}]'
            ],
            'AverageAgesOfOnset_df4': [
                "[{'AverageAgeOfOnset': 'Infancy'}]",
                "[{'AverageAgeOfOnset': 'Childhood'}]"
            ],
           'TypesOfInheritance_df4': [
                "[{'TypeOfInheritance': 'Autosomal dominant'}]",
                "[{'TypeOfInheritance': 'Autosomal recessive'}]"
           ],
            'PrevalenceData_df5': [
                '[{"PrevalenceClass": "1-9 / 100 000", "ValMoy": "4.0"}]',
                '[{"PrevalenceClass": "<1 / 1 000 000", "ValMoy": "0.0"}]'
            ],
             'SummaryInformation_df1': [
                '{"Definition": "This is <i>test</i> definition 1"}',
                '{"Definition": "This is test definition 2"}'
            ]
        }
        df = pd.DataFrame(data)
        df.to_csv(cls.test_file_path, index=False)

        # Initialize the data processor with the test directory
        cls.processor = OrphanetDataProcessor(data_dir=cls.test_data_dir)
        cls.processor.load_data("test_data.csv")


    @classmethod
    def tearDownClass(cls):
        # Clean up the temporary directory and file after tests
        os.remove(cls.test_file_path)
        os.rmdir(cls.test_data_dir)

    def test_load_data(self):
        self.assertIsNotNone(self.processor.data_df)
        self.assertEqual(len(self.processor.data_df), 2)

    def test_parse_hpo_associations(self):
        hpo_df = self.processor.parse_hpo_associations()
        self.assertIsNotNone(hpo_df)
        self.assertEqual(len(hpo_df), 2)
        self.assertEqual(hpo_df['HPOFrequencyValue'][0], 0.5)  # Check mapping
        self.assertEqual(hpo_df['HPOFrequencyValue'][1], 0.025)

    def test_parse_disability_associations(self):
        disability_df = self.processor.parse_disability_associations()
        self.assertIsNotNone(disability_df)
        self.assertEqual(len(disability_df), 2)
        self.assertEqual(disability_df['FrequencyDisabilityValue'][0], 0.5)
        self.assertEqual(disability_df['FrequencyDisabilityValue'][1], 0.025)

    def test_parse_average_age_of_onset(self):
        age_onset_df = self.processor.parse_average_age_of_onset()
        self.assertIsNotNone(age_onset_df)
        self.assertEqual(len(age_onset_df), 2)
        self.assertEqual(age_onset_df['AverageAgeOfOnset'][0], 'Infancy')
        self.assertEqual(age_onset_df['AverageAgeOfOnset'][1], 'Childhood')


    def test_parse_types_of_inheritance(self):
        inheritance_df = self.processor.parse_types_of_inheritance()
        self.assertIsNotNone(inheritance_df)
        self.assertEqual(len(inheritance_df), 2)
        self.assertEqual(inheritance_df['TypeOfInheritance'][0], 'Autosomal dominant')
        self.assertEqual(inheritance_df['TypeOfInheritance'][1], 'Autosomal recessive')

    def test_parse_prevalence_data(self):
        prevalence_df = self.processor.parse_prevalence_data()
        self.assertIsNotNone(prevalence_df)
        self.assertEqual(len(prevalence_df), 2)
        self.assertEqual(prevalence_df['PrevalenceClass'][0], "1-9 / 100 000")
        self.assertEqual(prevalence_df['PrevalenceClass'][1], "<1 / 1 000 000")

    def test_create_hpo_feature_matrix(self):
        hpo_df = self.processor.parse_hpo_associations()
        feature_matrix = self.processor.create_hpo_feature_matrix(hpo_df)
        self.assertIsNotNone(feature_matrix)
        self.assertIn('HP:0001', feature_matrix.columns)
        self.assertIn('HP:0002', feature_matrix.columns)

    def test_get_summary_information(self):
        summary_df = self.processor.get_summary_information()
        self.assertIsNotNone(summary_df)
        self.assertEqual(len(summary_df), 2)
        self.assertEqual(summary_df['Definition'][0], "This is test definition 1") # Check cleaning

    def test_prepare_data_for_ml(self):
        X, y = self.processor.prepare_data_for_ml()
        self.assertIsNotNone(X)
        self.assertIsNotNone(y)
        self.assertEqual(X.shape[0], y.shape[0]) #check length
        self.assertTrue(X.notna().all().all())  #check no NaN

    def test_get_hpo_term_mapping(self):
        hpo_mapping = self.processor.get_hpo_term_mapping()
        self.assertIsInstance(hpo_mapping, dict)
        self.assertEqual(hpo_mapping['HP:0001'], 'Term1')
        self.assertEqual(hpo_mapping['HP:0002'], 'Term2')

    def test_save_processed_data(self):
        output_path = os.path.join(self.test_data_dir, "processed_test_data.csv")
        # Ensure hpo_features is created
        self.processor.create_hpo_feature_matrix()
        self.processor.save_processed_data("processed_test_data.csv")
        self.assertTrue(os.path.exists(output_path))

        # Load the saved data and check if it's the same as the prepared data
        X, y = self.processor.prepare_data_for_ml()
        X['OrphaCode'] = y  # Add OrphaCode back for comparison
        saved_df = pd.read_csv(output_path)
        assert_frame_equal(X, saved_df, check_dtype = False)  # Use assert_frame_equal for DataFrame comparison. check_dtype = False because of prevalence
        os.remove(output_path)
    
    def test_safe_json_loads(self):
      good_json = '{"key": "value"}'
      bad_json = '{"key": "value"'
      none_json = None

      self.assertDictEqual(self.processor._safe_json_loads(good_json), {"key": "value"})
      self.assertIsNone(self.processor._safe_json_loads(bad_json))
      self.assertIsNone(self.processor._safe_json_loads(none_json))

    def test_extract_prevalence_value(self):
        self.assertEqual(self.processor._extract_prevalence_value('1-9 / 100 000'), (1+9)/2/100000)
        self.assertEqual(self.processor._extract_prevalence_value('<1 / 1 000 000'), 1/1000000)
        self.assertEqual(self.processor._extract_prevalence_value('5 / 1000'), 5/1000)
        self.assertEqual(self.processor._extract_prevalence_value('Unknown'), 0.0)
        self.assertEqual(self.processor._extract_prevalence_value('BadFormat'), 0.0)
        self.assertEqual(self.processor._extract_prevalence_value('N/A'), 0.0)



if __name__ == '__main__':
    unittest.main()