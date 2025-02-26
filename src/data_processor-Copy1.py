import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OrphanetDataProcessor:

    def __init__(self, data_dir: str = "../data"):
        self.data_dir = data_dir
        self.data_df = None
        self.hpo_features = None
        self.external_ref_features = None
        self.disability_features = None
        self.prevalence_features = None
        self.inheritance_features = None
        self.age_of_onset_features = None

        self.frequency_mapping = {
            "Very frequent (99-80%)": 0.9,
            "Frequent (79-30%)": 0.5,
            "Occasional (29-5%)": 0.15,
            "Very rare (<5%)": 0.025,
            "Excluded (0%)": 0.0,
            "": 0.0,  # Handle empty values
            "N/A": 0.0 # Handle N/A
        }

        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)

    def _safe_json_loads(self, json_string: str) -> Union[Dict, List, None]:
      """Safely loads potentially malformed JSON strings."""
      try:
          return json.loads(json_string)
      except (json.JSONDecodeError, TypeError):
          logger.warning(f"Invalid JSON encountered: {json_string}")
          return None # Or return an empty dict/list as appropriate
          
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Loads data from a CSV file."""
        full_path = os.path.join(self.data_dir, file_path)
        logger.info(f"Loading data from {full_path}")
        try:
            self.data_df = pd.read_csv(full_path)
            logger.info(f"Loaded {len(self.data_df)} records")
            return self.data_df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def parse_hpo_associations(self) -> pd.DataFrame:
      """Parses HPO associations from the HPODisorderAssociation_df2 column."""
      if self.data_df is None:
          raise ValueError("Data not loaded. Call load_data() first.")

      logger.info("Parsing HPO associations")
      hpo_data = []

      for idx, row in self.data_df.iterrows():
          orpha_code = row['OrphaCode']
          disease_name = row['Name']

          hpo_list = self._safe_json_loads(row['HPODisorderAssociation_df2'])
          if hpo_list is None:
            continue

          for hpo_item in hpo_list:
              hpo_data.append({
                  'OrphaCode': orpha_code,
                  'DiseaseName': disease_name,
                  'HPOId': hpo_item.get('HPOId', ''),
                  'HPOTerm': hpo_item.get('HPOTerm', ''),
                  'HPOFrequency': hpo_item.get('HPOFrequency', ''),
                  'HPOFrequencyValue': self.frequency_mapping.get(hpo_item.get('HPOFrequency', ''), 0.0),
                  'DiagnosticCriteria': hpo_item.get('DiagnosticCriteria', '')
              })

      hpo_df = pd.DataFrame(hpo_data)
      logger.info(f"Extracted {len(hpo_df)} HPO associations")
      return hpo_df
    def create_hpo_feature_matrix(self, hpo_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
      """Creates a feature matrix where each row is a disease and columns are HPO terms."""
      if hpo_df is None:
          hpo_df = self.parse_hpo_associations()

      logger.info("Creating HPO feature matrix")
      feature_matrix = hpo_df.pivot_table(
          index=['OrphaCode', 'DiseaseName'],
          columns='HPOId',
          values='HPOFrequencyValue',
          fill_value=0
      )
      feature_matrix = feature_matrix.reset_index()
      self.hpo_features = feature_matrix
      logger.info(f"Created feature matrix with {feature_matrix.shape[1]-2} HPO features")
      return feature_matrix

    def get_hpo_term_mapping(self, hpo_df: Optional[pd.DataFrame] = None) -> Dict[str, str]:
        """Creates a mapping from HPO IDs to HPO terms."""
        if hpo_df is None:
            hpo_df = self.parse_hpo_associations()
        hpo_mapping = hpo_df[['HPOId', 'HPOTerm']].drop_duplicates()
        hpo_dict = dict(zip(hpo_mapping['HPOId'], hpo_mapping['HPOTerm']))
        return hpo_dict

    def parse_disability_associations(self) -> pd.DataFrame:
      """Parses disability associations from DisabilityDisorderAssociations_df* columns."""
      if self.data_df is None:
          raise ValueError("Data not loaded. Call load_data() first.")

      logger.info("Parsing disability associations")
      disability_data = []

      for idx, row in self.data_df.iterrows():
          orpha_code = row['OrphaCode']
          disease_name = row['Name']
          #Iterating through the different disability columns
          for col_num in range(3, 6):  # Columns 3, 4, and 5
            col_name = f'DisabilityDisorderAssociations_df{col_num}'
            if col_name not in self.data_df.columns or pd.isna(row[col_name]):
                continue # Skip if column does not exist or if value is nan

            disability_list = self._safe_json_loads(row[col_name])
            if disability_list is None:
              continue
            for disability_item in disability_list:
              #Some entries does not have all the keys, so it's safer to use .get with defaults
              disability_data.append({
                  'OrphaCode': orpha_code,
                  'DiseaseName': disease_name,
                  'Disability': disability_item.get('Disability', ''),
                  'FrequencyDisability': disability_item.get('FrequencyDisability', ''),
                  'FrequencyDisabilityValue': self.frequency_mapping.get(disability_item.get('FrequencyDisability', ''), 0.0),
                  'TemporalityDisability': disability_item.get('TemporalityDisability', ''),
                  'SeverityDisability': disability_item.get('SeverityDisability', ''),
                  'LossOfAbility': disability_item.get('LossOfAbility', ''),
                  'TypeDisability': disability_item.get('TypeDisability', ''),
                  'Defined': disability_item.get('Defined', '')
                })

      disability_df = pd.DataFrame(disability_data)
      logger.info(f"Extracted {len(disability_df)} disability associations")
      return disability_df

    def parse_average_age_of_onset(self) -> pd.DataFrame:
      """Parses average age of onset from AverageAgesOfOnset_df* columns."""
      if self.data_df is None:
          raise ValueError("Data not loaded.  Call load_data() first.")
      logger.info("Parsing average age of onset")
      age_of_onset_data = []

      for idx, row in self.data_df.iterrows():
          orpha_code = row['OrphaCode']
          disease_name = row['Name']
          for col_num in range(4, 6):
            col_name = f'AverageAgesOfOnset_df{col_num}'
            if col_name not in self.data_df.columns or pd.isna(row[col_name]):
                continue
            age_of_onset_list = self._safe_json_loads(row[col_name])
            if age_of_onset_list is None:
                continue
            
            for age_item in age_of_onset_list:
              age_of_onset_data.append({
                  'OrphaCode': orpha_code,
                  'DiseaseName': disease_name,
                  'AverageAgeOfOnset': age_item.get('AverageAgeOfOnset', '') #Using .get()
              })

      age_of_onset_df = pd.DataFrame(age_of_onset_data)
      logger.info(f"Extracted {len(age_of_onset_df)} age of onset entries")
      return age_of_onset_df
        
    def parse_types_of_inheritance(self) -> pd.DataFrame:
      if self.data_df is None:
          raise ValueError("Data not loaded.  Call load_data() first.")

      logger.info("Parsing types of inheritance")
      inheritance_data = []
      for idx, row in self.data_df.iterrows():
        orpha_code = row['OrphaCode']
        disease_name = row['Name']

        for col_num in range(4, 6):
          col_name = f'TypesOfInheritance_df{col_num}'
          if col_name not in self.data_df.columns or pd.isna(row[col_name]):
            continue
          inheritance_list = self._safe_json_loads(row[col_name])
          if inheritance_list is None:
            continue

          for inheritance_item in inheritance_list:
            inheritance_data.append({
              'OrphaCode': orpha_code,
              'DiseaseName': disease_name,
              'TypeOfInheritance': inheritance_item.get('TypeOfInheritance', '')
            })
      inheritance_df = pd.DataFrame(inheritance_data)
      logger.info(f"Extracted {len(inheritance_df)} inheritance entries")
      return inheritance_df

    def parse_prevalence_data(self) -> pd.DataFrame:
        """Parses prevalence data from the PrevalenceData_df5 column."""
        if self.data_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        logger.info("Parsing prevalence data")
        prevalence_data = []

        for idx, row in self.data_df.iterrows():
            orpha_code = row['OrphaCode']
            disease_name = row['Name']

            prevalence_list = self._safe_json_loads(row['PrevalenceData_df5'])
            if prevalence_list is None:
                continue

            for prev_item in prevalence_list:
                prevalence_data.append({
                    'OrphaCode': orpha_code,
                    'DiseaseName': disease_name,
                    'PrevalenceType': prev_item.get('PrevalenceType', ''),
                    'PrevalenceQualification': prev_item.get('PrevalenceQualification', ''),
                    'PrevalenceClass': prev_item.get('PrevalenceClass', ''),
                    'ValMoy': float(prev_item.get('ValMoy', 0.0)),  # Convert to float
                    'PrevalenceGeographic': prev_item.get('PrevalenceGeographic', ''),
                    'PrevalenceValidationStatus': prev_item.get('PrevalenceValidationStatus', '')
                })

        prevalence_df = pd.DataFrame(prevalence_data)
        logger.info(f"Extracted {len(prevalence_df)} prevalence entries")
        return prevalence_df


    def get_summary_information(self) -> pd.DataFrame:
        if self.data_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        logger.info("Extracting summary information")
        summary_data = []
        for idx, row in self.data_df.iterrows():
          orpha_code = row['OrphaCode']
          disease_name = row['Name']

          summary_info = self._safe_json_loads(row['SummaryInformation_df1'])
          if summary_info is None:
            continue

          definition = summary_info.get('Definition', '')
          definition = definition.replace('<i>', '').replace('</i>', '')  # Clean HTML

          summary_data.append({
              'OrphaCode': orpha_code,
              'DiseaseName': disease_name,
              'Definition': definition
          })

        summary_df = pd.DataFrame(summary_data)
        logger.info(f"Extracted {len(summary_df)} disease summaries")
        return summary_df
    
    def _extract_prevalence_value(self, prevalence_class: str) -> float:
      """Extracts a numerical prevalence value from the PrevalenceClass string."""
      #Handles prevalence strings, like '<1 / 1 000 000'  or '1-5 / 10 000'
      if prevalence_class == 'Unknown' or  prevalence_class == 'N/A':
        return 0.0
      parts = prevalence_class.split('/')
      if len(parts) != 2:
        return 0.0  # Handle malformed strings
      try:
        numerator_str = parts[0].strip()
        denominator_str = parts[1].strip().replace(' ', '') # Remove spaces in denominator

        # Handle ranges in numerator
        if '-' in numerator_str:
            numerator_range = [float(x) for x in numerator_str.split('-')]
            numerator = sum(numerator_range) / len(numerator_range) # Average the range
        else:
            numerator = float(numerator_str)

        denominator = float(denominator_str)
        return numerator/denominator
      except ValueError:
        logger.warning(f'Couldnt extract a value from: {prevalence_class}')
        return 0.0 # Handle parsing errors

    def prepare_data_for_ml(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepares the final dataset for machine learning."""
        logger.info("Preparing final dataset for machine learning")

        # Get HPO features
        if self.hpo_features is None:
            self.create_hpo_feature_matrix()

        # Create disability features (pivot as needed)
        disability_df = self.parse_disability_associations()
        if not disability_df.empty:
            self.disability_features = disability_df.pivot_table(
                index='OrphaCode',
                columns='Disability',
                values='FrequencyDisabilityValue',
                fill_value=0
            ).reset_index()


        # Create age of onset features (pivot as needed)
        age_of_onset_df = self.parse_average_age_of_onset()
        if not age_of_onset_df.empty:
            self.age_of_onset_features = age_of_onset_df.pivot_table(
                index='OrphaCode',
                columns='AverageAgeOfOnset',
                values = 'DiseaseName', #dummy value
                aggfunc = 'first', #Presence/absence
                fill_value=0
              ).reset_index()
            #Flatten MultiIndex
            self.age_of_onset_features.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in self.age_of_onset_features.columns]

        # Create inheritance features
        inheritance_df = self.parse_types_of_inheritance()
        if not inheritance_df.empty:
            self.inheritance_features = inheritance_df.pivot_table(
                index='OrphaCode',
                columns='TypeOfInheritance',
                values = 'DiseaseName', #dummy value
                aggfunc = 'first',
                fill_value = 0
            ).reset_index()

        # Create prevalence features - keep raw values and class.
        prevalence_df = self.parse_prevalence_data()
        if not prevalence_df.empty:
            # Numerical prevalence.
            prevalence_df['PrevalenceValue'] = prevalence_df['PrevalenceClass'].apply(self._extract_prevalence_value)
            self.prevalence_features = prevalence_df.groupby('OrphaCode')['PrevalenceValue'].mean().reset_index() # Use mean for now

        # Merge features
        features = self.hpo_features
        if self.disability_features is not None:
          features = features.merge(self.disability_features, on='OrphaCode', how='left')
        if self.age_of_onset_features is not None:
          features = features.merge(self.age_of_onset_features, on='OrphaCode', how='left')
        if self.inheritance_features is not None:
            features = features.merge(self.inheritance_features, on='OrphaCode', how='left')
        if self.prevalence_features is not None:
            features = features.merge(self.prevalence_features, on='OrphaCode', how='left')


        # Extract target and features
        y = features['OrphaCode']
        X = features.drop(['OrphaCode', 'DiseaseName'], axis=1, errors='ignore')

        # Fill missing values (left after merging)
        X = X.fillna(0)

        # Apply prevalence weighting (example - adjust as needed)
        # This is a *simple* example.  You might want more sophisticated weighting.
        if 'PrevalenceValue' in X.columns:
            # Scale prevalence to avoid overwhelming other features
            #  Here, we're simply taking the log, but you might want something else.
            X['PrevalenceValue'] = np.log1p(X['PrevalenceValue'])
        logger.info(f"Final dataset: {X.shape[0]} samples, {X.shape[1]} features")

        return X, y
    
    def save_processed_data(self, output_path: str) -> None:
        """Save processed data to CSV file."""
        if self.hpo_features is None:
            raise ValueError("No processed data available.  Process data first.")

        full_path = os.path.join(self.data_dir, output_path)
        logger.info(f"Saving processed data to {full_path}")

        X, y = self.prepare_data_for_ml()
        output_df = X.copy()
        output_df['OrphaCode'] = y  # Add target back
        output_df.to_csv(full_path, index=False)
        logger.info(f"Saved processed data with {output_df.shape[1]} columns")