"""
Copyright 2017 Rahul Gupta, Soham Pal, Aditya Kanade, Shirish Shevade.
Indian Institute of Science.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse, time
import numpy as np
import os, sys

from process import build_dictionary, generate_binned_training_data, save_dictionaries, save_test_problems, vectorize_data, save_folds
from util.helpers import done

parser = argparse.ArgumentParser(description="Process 'C' dataset to be used in repair tasks")

parser.add_argument("-t", "--task", help="specify the task", choices=['typo', 'ids'], default='typo')
parser.add_argument("--dataset", help="specify the task", choices=['iitk', 'codechef'], default='iitk')
parser.add_argument("--max_prog_length", type=int, help="maximum length of the programs in tokens", default=400)
parser.add_argument("--min_prog_length", type=int, help="minimum length of the programs in tokens", default=10)
parser.add_argument("--max_fix_length", type=int, help="maximum length of the fixes in tokens", default=20)
parser.add_argument("--max_mutations", type=int, help="maximum mutations per program", default=5)
parser.add_argument("--num_variants", type=int, help="maximum pairs per program", default=2)

args = parser.parse_args()

if args.task == 'typo':
    kind_mutations = 'typo'    
else:
    kind_mutations = 'ids'

dataset = args.dataset

output_directory   = os.path.join('network_inputs')

print"data_generator.py  passed" #debug
    
try:
    os.makedirs(output_directory)
except:
    pass

# Limits
max_program_length = args.max_prog_length
min_program_length = args.min_prog_length
max_fix_length     = args.max_fix_length

# Mutations
max_mutations      = args.max_mutations       # per corrupted copy
num_variants       = args.num_variants        # number of corrupted copies

if kind_mutations == 'typo':
    drop_ids = True
    mutations_series = True
    print 'Dropping IDs...'
else:
    drop_ids = False
    mutations_series = False
    print 'Keeping IDs...'    

print ''

print"data_generator.py  Gnerate data"
# Generate data
token_strings, mutations_distribution, program_lists = generate_binned_training_data(\
                     max_program_length, min_program_length, max_fix_length,\
                     kind_mutations, max_mutations, num_variants, mutations_series)

print"data_generator.py  Build the dictionary"
# Build the dictionary
tl_dict, rev_tl_dict = build_dictionary(token_strings, drop_ids)

print"data_generator.py  Tokenize"
# Tokenize
token_vectors = vectorize_data(token_strings, tl_dict, max_program_length, max_fix_length, drop_ids)

print"data_generator.py  Save everything"
# Save everything
save_folds(output_directory, token_vectors)

save_dictionaries(output_directory, tl_dict, rev_tl_dict)
np.save(os.path.join(output_directory, 'training-error-distribution.npy'), mutations_distribution)
np.save(os.path.join(output_directory, 'program_lists.npy'), program_lists)

#######
'''
training_error_dist = np.load(os.path.join('network_inputs','training-error-distribution.npy')).item()
program_lists = np.load(os.path.join('network_inputs','program_lists.npy')).item()
#ex-conf = np.load(os.path.join('network_inputs','experiment-configuration.npy')).item()
rev_translate_dict = np.load(os.path.join('network_inputs','rev_translate_dict.npy')).item()
translate_dict = np.load(os.path.join('network_inputs','translate_dict.npy')).item()
print "program_lists = ",program_lists
print "training_error_distoributuion = ",training_error_dist
print "translate_dict = ",translate_dict
print "rev_translate_dict = ",rev_translate_dict
#print "ex-conf = ",ex-conf
'''
#######

configuration = {}

configuration["args"]             = args
configuration["drop_ids"]         = drop_ids
configuration["mutations_series"] = mutations_series

np.save(os.path.join(output_directory, 'experiment-configuration.npy'), configuration)
print"data_generator.py  pre-dpne()"

done()
