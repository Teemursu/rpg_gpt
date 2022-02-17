

import ast
import pandas as pd
from pprint import pprint

df = pd.read_csv('data\persuasion_dataset.csv', encoding='utf-8', sep="\t")
#df = df[df['filename'] == "extracting_data/infinity_engine/kotor1/xml/22aa_zaalb01_01.dlg.xml"]

#context_df = pd.Dataframe(columns=['response', 'context/0', 'context/1','context/2','context/3','context/4','context/5','context/6'])
context_df = pd.DataFrame()

def get_context_row(i, row):
  global df
  global context_df
  lst_games = ['KotOR 1', 'KotOR 2', 'NWN 1']
  filename = current_row['filename']
  conversation = df[df['filename'] == filename]
  reply_id = ""

  if row['game'] in lst_games:
    replylist = ast.literal_eval(row['replies'])
    for reply_id in replylist:
      context_list = []

      if reply_id != "EoC":
        if row['persuasion'] != 'False':
          persrow = "<{}>".format(row['persuasion']) + " {}".format(row['en'])
        else:
          persrow= row['en']
        #if row['en'].shape == (1,):
        #  context_list.append(row['en'].item())
        #else:
        #  context_list.append(row['en'].iloc[0])

        reply = conversation[conversation['sid'] == float(reply_id)]
        if reply.shape != (0,8):
          context_list.append(persrow)
          index = reply.index
          #print("replyid is",reply_id,"from",replylist)
          if reply['replies'].shape == (1,):
            replylist = reply.loc[index, 'replies'].item()
          else:
            replylist = reply['replies'].iloc[0]

          replylist = ast.literal_eval(replylist)
          #print("replyid is",reply_id,"from",replylist)
          if reply['en'].shape == (1,):
            if reply['persuasion'].item() != "False":
              persreply = "<{}>".format(reply['persuasion'].item()) + " {}".format(reply['en'].item())
            else:
              persreply = reply['en'].item()
            
            context_list.append(persreply)
          else:
            if reply['persuasion'].iloc[0] != "False":
              persreply = "<{}>".format(reply['persuasion'].iloc[0]) + " {}".format(reply['en'].iloc[0])
            else:
              persreply = reply['en'].iloc[0]
            
            context_list.append(persreply)

          while replylist != []:
            for reply_id in replylist:
              replylist.remove(reply_id)

              if reply_id != "EoC":
            
                reply = conversation[conversation['sid'] == float(reply_id)]
                if reply.shape != (0,8):
                # replyid is 10436 from ['10436', '10435']
                # .. should then '10402' from ['10402']
                  #print("replyid is",reply_id,"from",replylist)
                  
                  if reply['en'].shape == (1,):
                    if reply['persuasion'].item() != "False":
                      persreply = "<{}>".format(reply['persuasion'].item()) + " {}".format(reply['en'].item())
    
                    else:
                      persreply = reply['en'].item()
                    
                    context_list.append(persreply)
                  else:
                    if reply['persuasion'].iloc[0] != "False":
                      persreply = "<{}>".format(reply['persuasion'].iloc[0]) + " {}".format(reply['en'].iloc[0])

                    else:
                      persreply = reply['en'].iloc[0]
              
                    context_list.append(persreply)

            print()
            for elem in context_list:
              print("-",elem)
          #print(context_list)
          #print()
          context_list.reverse()
          temp_df = pd.DataFrame([context_list], columns=['response'] + ["context/{}".format(i) for i in range(len(context_list)-1)],
                                )
          #print(temp_df)
          context_df = pd.concat([temp_df, context_df])
          #print(context_df.keys())

                #print("replyid changed to", replylist)


for i, current_row in df.iterrows():
  context_row = get_context_row(i, current_row)




context_path = 'NLG_dataset.csv'
context_df.to_csv(context_path, encoding="utf-8", sep="\t")