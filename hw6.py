from surprise import KNNBasic
from surprise import NMF
from surprise import SVD
from surprise import Dataset
from surprise import evaluate, print_perf
import os
from surprise import Reader


#load data from a file 
file_path = os.path.expanduser('restaurant_ratings.txt')
reader = Reader(line_format='user item rating timestamp', sep='\t')
data = Dataset.load_from_file(file_path, reader=reader)

#split data
data.split(n_folds=3)

#PART 5
print("\n\n\n")
algo = SVD() #SVD
perf = evaluate(algo,data,measures=['RMSE','MAE'])
print_perf(perf)


#PART 6
print("\n\n\n")
algo = SVD(biased=False) #PMF
perf = evaluate(algo, data,measures=['RMSE', 'MAE'])
print_perf(perf)


#PART 7
print("\n\n\n")
algo = NMF() #NMF
perf = evaluate(algo, data,measures=['RMSE', 'MAE'])
print_perf(perf)


#PART 8
print("\n\n\n")
algo = KNNBasic(sim_options = {'user_based': True }) #user based collaborative filtering
perf = evaluate(algo, data,measures=['RMSE', 'MAE'])
print_perf(perf)


#PART 9
print("\n\n\n")
algo = KNNBasic(sim_options = {'user_based': False }) #item based collaborative filtering
perf = evaluate(algo, data,measures=['RMSE', 'MAE'])
print_perf(perf)


#part 14
print("\n\n\n")
algo = algo = KNNBasic(sim_options = {'name':'MSD','user_based': True }) #user based collaborative filtering
perf = evaluate(algo, data,measures=['RMSE', 'MAE'])
print_perf(perf)

print("\n\n\n")
algo = algo = KNNBasic(sim_options = {'name':'cosine','user_based': True }) #user based collaborative filtering
perf = evaluate(algo, data,measures=['RMSE', 'MAE'])
print_perf(perf)

print("\n\n\n")
algo = algo = KNNBasic(sim_options = {'name':'pearson','user_based': True }) #user based collaborative filtering
perf = evaluate(algo, data,measures=['RMSE', 'MAE'])
print_perf(perf)

print("\n\n\n")
algo = algo = KNNBasic(sim_options = {'name':'MSD','user_based': False }) #user based collaborative filtering
perf = evaluate(algo, data,measures=['RMSE', 'MAE'])
print_perf(perf)

print("\n\n\n")
algo = algo = KNNBasic(sim_options = {'name':'cosine','user_based': False }) #user based collaborative filtering
perf = evaluate(algo, data,measures=['RMSE', 'MAE'])
print_perf(perf)

print("\n\n\n")
algo = algo = KNNBasic(sim_options = {'name':'pearson','user_based': False }) #user based collaborative filtering
perf = evaluate(algo, data,measures=['RMSE', 'MAE'])
print_perf(perf)



