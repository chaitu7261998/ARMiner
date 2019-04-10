from topic_modeling import model_topics
from helper import get_rating
import numpy as np
import preprocess

def group_volume(review_group_matrix):
	return (review_group_matrix.sum(axis=0))

def group_average_rating(review_group_matrix, volumes, ratings):
	group_count = volumes.shape[0]
	average_ratings = np.array([(volumes[g] / np.dot(review_group_matrix[:, g], ratings)) for g in range(group_count)])
	return average_ratings

def get_group_rankings(review_group_matrix, ratings,group_weights):
	volumes = group_volume(review_group_matrix)
	print(volumes)
	average_ratings = group_average_rating(review_group_matrix, volumes, ratings)
	group_count = volumes.shape[0]
	group_scores = [ (group_weights[0] * volumes[idx] + group_weights[1]*average_ratings[idx],idx+1) for idx in range(group_count)]
	group_scores.sort()
	rankings = [ group_scores[idx][1] for idx in range(group_count)]
	return rankings

def review_ratings(informative_reviews):
	ratings = np.zeros(informative_reviews.shape[0])
	for idx, instance in enumerate(informative_reviews) :
		ratings[idx] = get_rating(instance)
		if ratings[idx] == -1 :
			return ("No rating allocated")
	return ratings

def get_group_weights(groups):
	return np.full(groups, 1/groups)

def get_instance_weights(instances):
	return np.full(instances, 1/instances)

def jaccard_sim(instance1, instance2):
    count_intersect = 0
    count_1 = 0
    count_2 = 0
    for i in range(0,len(instance1)):
        if instance1[i] == 1:
            count_1 += 1
        if instance2[i] == 1:
            count_2 += 1
        count_intersect += int((instance1[i] + instance2[i])/2)

    #print(float(count_intersect/(count_1 + count_2 - count_intersect)))
    return float(count_intersect/(count_1 + count_2 - count_intersect))

def group_ranking(useful_data, review_group_matrix, group_number):
    print("Hello")
    is_duplicate = [0]*len(useful_data)
    unique_useful_data = []
    unique_data_count = []
    unique_data_proportion = []

    print("length: ",len(useful_data))

    similarity_cutoff = 0.8
    for i in range(0,len(useful_data)):
        if is_duplicate[i] != 0:
            continue
        data_count = 1
        tmp = useful_data[i]
        rating = get_rating(useful_data[i])
        prop = review_group_matrix[i][group_number]
        for j in range(0,len(useful_data)):
            #print("J: ",j)
            #print("GN: ",group_number)

            if jaccard_sim(useful_data[i],useful_data[j]) >= similarity_cutoff:
                is_duplicate[j] = 1
                data_count += 1
                prop = max(prop, review_group_matrix[j][group_number])

                if get_rating(useful_data[j]) < rating:
                    for r in range(6):
                        tmp[r] = 0
                    tmp[get_rating(useful_data[j])] = 1

        unique_data_proportion.append(prop)
        unique_useful_data.append(tmp)
        unique_data_count.append(data_count)

    # Give Rating



def main():

    training_data_list = ["datasets/swiftkey/trainL/info.txt","datasets/swiftkey/trainL/non-info.txt"]
    training_data_info = "datasets/swiftkey/trainL/info.txt"
    training_data_noninfo = "datasets/swiftkey/trainL/non-info.txt"
    test_data = "datasets/swiftkey/trainU/unlabeled.txt"

    review_group_matrix, useful_data, mapping = model_topics(training_data_list,
    														 training_data_info,
    								 						 training_data_noninfo,
    														 test_data)

    ratings = review_ratings (useful_data)

    instance_weights = get_instance_weights(review_group_matrix.shape[0])
    group_weights = get_group_weights(2)
    group_rankings = get_group_rankings(review_group_matrix, ratings, group_weights)
    print(group_rankings)

    num_of_groups = len(review_group_matrix[0])

    groups = [[] for i in range(num_of_groups)]

    group_cutoff = 0.01

    for i in range(0,len(useful_data)):
        j = 0
        for j in range(0,num_of_groups):
            if review_group_matrix[i][j] >= group_cutoff:
                groups[j].append(useful_data[i])

    group_ranking_result = []
    for i in range(len(groups)-9):
        group_ranking_result.append(group_ranking(groups[i], review_group_matrix, i))

    # is_duplicate = [0]*len(useful_data)
    # unique_useful_data = []
    # unique_data_count = []
    #
    # similarity_cutoff = 0.8
    # for i in range(0,len(useful_data)):
    #     if is_duplicate[i] != 0:
    #         continue
    #     data_count = 1
    #     tmp = useful_data[i]
    #     rating = get_rating(useful_data[i])
    #     for j in range(0,len(useful_data)):
    #
    #
    #         if jaccard_sim(useful_data[i],useful_data[j]) >= similarity_cutoff:
    #             is_duplicate[j] = 1
    #             data_count += 1
    #             if get_rating(useful_data[j]) < rating:
    #                 for r in range(6):
    #                     tmp[r] = 0
    #                 tmp[get_rating(useful_data[j])] = 1
    #
    #             # Set Probability
    #     unique_useful_data.append(tmp)
    #     unique_data_count.append(data_count)
    #
    # print(len(unique_useful_data))



if __name__ == '__main__':
	main()
