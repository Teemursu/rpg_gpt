import pandas as pd
import xmltodict
import json
from pprint import pprint

kotor_df = pd.read_csv('data/kotor_dataset.csv')
nwn_df = pd.read_csv('data/nwn_dataset.csv')

kotor1_tlk = 'extracting_data/infinity_engine/kotor1/kotor1_en.xml'
kotor2_tlk = 'extracting_data/infinity_engine/kotor2/kotor2_en.xml'
files = [kotor1_tlk, kotor2_tlk]
for file in files:
    with open(file, 'rb') as f:
            xml = xmltodict.parse(f)
            xml = json.dumps(xml)
            xml = json.loads(xml)
            for elem in xml['tlk']['string']:
                print(elem)