import pandas as pd
import xmltodict
import json
from pprint import pprint
import numpy as np
import re
from pprint import pprint

perses = ["[Persu", ';Persu',
          "berreden]", "berreden&"
          "[Überre", ";Überreden", "[ Überre",
          '[Peru', ';Peru', "[ Persu", '[ Peru',
          '[Überreden/Lügen]', '[Überreden/Bluff]']

kotor_df = pd.read_csv('data/kotor_dataset.csv')
nwn_df = pd.read_csv('data/nwn_dataset.csv')
df = pd.concat([kotor_df, nwn_df], axis=0)

df['sentence'] = np.nan
df['is_persuasion'] = np.nan
df['sid'] = pd.to_numeric(df['sid'], downcast='integer')
df.set_index('sid', inplace=True)
kotor1_tlk = 'extracting_data/infinity_engine/kotor1/tlk/kotor1_en.xml'
kotor2_tlk = 'extracting_data/infinity_engine/kotor2/tlk/kotor2_en.xml'
nwn_tlk = 'extracting_data/infinity_engine/nwn/tlk/nwn_en.xml'
files = [kotor1_tlk, kotor2_tlk, nwn_tlk]

df_games_dict = {game: df[df['game'] == game] for game in df.game.unique()}
#print(df_games_dict)
games = df_games_dict.keys()
pprint(games)
for file, game in zip(files, games):
        print(df)
        df = df_games_dict[game]
        with open(file, 'rb') as f:
                xml = xmltodict.parse(f)
                xml = json.dumps(xml)
                xml = json.loads(xml)
                for elem in xml['tlk']['string']:
                    #print(df['sid'][253])
                    try:
                        #df.loc[(df['game'] == game) & (df.index == int(elem['@id'])), 'sentence'] = elem['#text']
                        df.loc[df.index == np.intc(elem['@id']), 'sentence'] = elem['#text']
                        
                        if any(map(elem['#text'].__contains__, perses)):
                            split_pers = elem['#text'].split("]")
                            pers_tag = re.sub("\[", "", split_pers[0])
                            pers_tag = re.sub("<StartCheck>", "", pers_tag)
                            pers_tag = re.sub("</Start>", "", pers_tag)
                            pers_tag = re.sub(" ", "", pers_tag)
                            pers_tag = re.sub(r"[\/\,\-\+]", "/", pers_tag)
                            
                            df.loc[df.index == np.intc(elem['@id']), 'persuasion'] = pers_tag
                            if len(df.loc[(df.index == np.intc(elem['@id']))]) == 0 and game=='KotOR 1':
                                print(elem['@id'], elem['#text'])
                            
                            
                    except KeyError:
                        pass
        df_games_dict[game] = df

df = pd.concat([df for df in df_games_dict.values()], axis=0,)
filename = 'persuasion_dialogue_dataset.csv'
df.to_csv(filename, encoding='utf-8')
