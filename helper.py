import numpy as np
# helper function takes in the data and word mappings 
# and returns the words corresponding to each instance for the data

def get_all_instance_words(reverse_mapping,data,file_name):

	# word id starts from zero
	rating_bits = 6
	instance_count = 1
	
	word_list = open(file_name,"w")

	# prints the instances and the words per instance for reference 
	for line in data:		
		word_id = 0
		line = line[rating_bits:]
		meta="------Rating Instance:"+str(instance_count)+"--------\n"
		word_list.write(meta)
		for word_id in range(len(line)):
			if( line[word_id] != 0 ):
				word_list.write( reverse_mapping[word_id]+"\n" )
		instance_count = instance_count + 1
	word_list.close()

def get_instance_words(reverse_mapping,data):
	# word id starts from zero
	rating_bits = 6
	instance_count = 1

	for v in data :
		print(v, end=', ')
	# print(data)
	indices = np.nonzero(data)[0][1:]
	print(indices)
	for index in indices:
		print(reverse_mapping[index-rating_bits])
	