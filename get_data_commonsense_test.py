import pandas as pd
import os
from sklearn.model_selection import train_test_split
import pickle
from collections import OrderedDict

data_dir = "/home/rad/Documents/sbu_acads/thesis/common_sense/dataset/storycommonsense_data/csv_version/test/emotion/"
data = pd.read_csv(os.path.join(data_dir, 'allcharlinepairs.csv'))
# print data.head

# data_subset = data[['storyid', 'linenum', 'char', 'context', 'sentence', 'emotion', 'plutchik']]
# print data_subset.head

#get unique storyids
storyids = list(set(data['storyid']))

#use all the point in the test file for testing the model
storyid_test = storyids
# print "Storyids are:"

# print len(storyids)

#splitting data into training and test set based on the storyids
# storyid_train, storyid_test = train_test_split(storyids, test_size = 0.2, train_size = 0.8, shuffle=True)

# print "Train data:"
# print len(storyid_train)
# print "Test data:"
# print len(storyid_test)

#preparing data for encoding
#storyid: {label : [s1, context c1]}


# def get_input_data(storyid):
	# pass

#dictionary to hold the data in the format to be input into the encoder
data_dictonary = dict()
# data_dictonary_val = dict()

for s_id in storyid_test:
	storyid_true = data['storyid'] == s_id
	df_storyid = data[storyid_true]
	# print "Df for story id s_id"
	# print df_storyid
	# print 

	groupby_char_line = df_storyid.groupby(['linenum', 'char', 'emotionworkerid'])
	linecount = 0

	for name, group in groupby_char_line:
		print name
		# print
		print group

		linenum = name[0]
		char = name[1]
		print "Line num:"
		print linenum
		print "Char"
		print char
		print "Annotator"
		print name[2]
		char_lower = char.lower()

		if linenum == 1:
			print "No context present"
			context_string = ""   #add an empty context string
		else:
			# context_string = group["context"]
			#to get the context for the particular character appearing in the line
			context_line_num_list = [i for i in range(1, linenum)]
			print "context_line_num_list are:"
			print context_line_num_list
			print

			#get context lines for the characters
			context_df = df_storyid[(df_storyid['linenum'] >= context_line_num_list[0]) & (df_storyid['linenum'] <= context_line_num_list[-1]) & (df_storyid['char'] == char)]
			print "context df is:"
			print context_df[["char", "sentence"]]
			# print
			#remove repetitions while maintaining order
			context_list = list(OrderedDict.fromkeys(context_df["sentence"].values))
			print context_list
			#reverse the order of the context so that the oldest is first
			#and the most recent one is last
			context_string = (" ".join(context_list)).strip()
			print context_string
			print

		try:
			for label in group['plutchik']:
				label_str_list = label.split(":")
				label_clean = label_str_list[0].strip('[')
				label_clean = label_clean.strip(']')
				label_clean = label_clean.strip('"')
				"Entry to be added is:"
				print {label_clean : [group['sentence'].values[0], context_string]}
				print
				data_dictonary[s_id].append({label_clean : [group['sentence'].values[0], context_string]})

		except Exception, e:
				data_dictonary[s_id] = [{label_clean : [group['sentence'].values[0], context_string]}]	
			
		# linecount += 1 
		print "###############################################################"
		print
		# if linecount > 10:
			# break

	# break

print "The final data dictionary is:"
print data_dictonary

pickle.dump(data_dictonary, open("test_data.p", "wb"))
