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

import numpy as np
import tensorflow as tf
import os, tempfile
from os import path
import argparse
import subprocess32 as subprocess
from util.tokenizer import C_Tokenizer
from util.helpers import tokens_to_source, compilation_errors, fix_to_source, extract_line_number
from neural_net.model import model
from util.postprocessing_helpers import vectorize, devectorize, apply_fix, SubstitutionFailedException, InvalidFixLocationException, VectorizationFailedException

def get_fix(sess, program, network):
    print "debug--1"
    X = np.reshape(program, (1, len(program)))
    print "debug--2"
    return network['seq2seq'].sample(sess, X)[0]

def get_fixes(sess, programs, network):
        return network['seq2seq'].sample(sess, programs)

class DeepFixServer:
    def __init__(self, fold, embedding_dim, memory_dim, num_layers, rnn_cell, dropout, task, vram_fraction):
        training_args = np.load(os.path.join('network_inputs', 'experiment-configuration.npy')).item()['args']
        dictionary = np.load(os.path.join('network_inputs', 'translate_dict.npy')).item()
	print training_args
	print dictionary
	print "1..myself"
        self.network = {
            # Figure out stuff
            'training_args': training_args,
            'dictionary': dictionary,

            'in_seq_length': training_args.max_prog_length,
            'out_seq_length': training_args.max_fix_length,
            'vocab_size': len(dictionary),

            # Build the network
            'seq2seq': model(training_args.max_prog_length, training_args.max_fix_length, len(dictionary),
                            rnn_cell=rnn_cell, memory_dim=memory_dim, num_layers=num_layers, dropout=dropout,
                            embedding_dim=embedding_dim, bidirectional=False, trainable=False)
        }
	print "3..myself"
        self.task = task
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=vram_fraction)
	print "4..myself"
        self.network['session'] = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
	print "5..myself"
        checkpoint_to_load = os.path.join('checkpoints', 'fold_%d' % fold,'best')
	print "6..myself"
        checkpoint_to_load = os.path.join(checkpoint_to_load, os.listdir(checkpoint_to_load)[0])
        print "7..myself"
#	pathabs = os.path.abspath(tf.train.latest_checkpoint(checkpoint_to_load))
	print checkpoint_to_load
#	ckpt-load = tf.train.latest_checkpoint(checkpoint_to_load)
#	print ckpt-load
	print self.network['session'] 
#	ckpt=tf.train.get_checkpoint_state(pathabs)
#	if ckpt:
#		print "7.1..myself"
	#	last_model=ckpt.model_checkpoint_path
	self.network['seq2seq'].load_parameters(self.network['session'],checkpoint_to_load)
#	else:
#		print "7.2..myself"
#		init=tf.initialize_all_variables()
#		self.network['session'].run(init)
	print "8..myself"
######	self.network['session'].close()
    def process(self, source_code_array, max_attempts=6):
        sequences_of_programs = {}
        fixes_suggested_by_network = {}
        entries = []
        entries_ids = []
        errors = {}
        fixes_to_return = {}
        error_messages = {}

        print "----------------------"
        print source_code_array
        print "----------------------"
        # Wrap it up into a nice box
        for idx, source_code in enumerate(source_code_array):
            program, name_dict, name_sequence, literal_sequence = C_Tokenizer().tokenize(source_code)
            entries.append((idx, program, name_dict, name_sequence, literal_sequence))
            entries_ids.append((idx, program, name_dict, name_sequence, literal_sequence))
            sequences_of_programs[idx] = [program]
            fixes_suggested_by_network[idx] = []
            errors[idx], _ = compilation_errors(source_code)
	    print "debug..1"
            error_messages[idx] = []
	    print "debug..2"
            fixes_to_return[idx] = []
	    print "debug..3"

        network = self.network
	print "debug..4"

        if self.task == 'ids':
            normalize_names = False
            fix_kind = 'insert'
            
        else:
            assert self.task == 'typo'
            normalize_names = True
            fix_kind = 'replace'

        # Reinitialize `entries'
	print "debug..5"
        entries = entries_ids

        try:
            for round_ in range(max_attempts):
		print "debug..6"
                to_delete = []
                input_ = []

                for i, entry in enumerate(entries):
		    print "debug..7"
                    idx, program, name_dict, name_sequence, literal_sequence = entry

                    try:
			print "debug..8"
                        program_vector = vectorize(sequences_of_programs[idx][-1], network['in_seq_length'], network['dictionary'], normalize_names=normalize_names, reverse=True, append_eos=False)

			'''
		    	print "]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]"
		    	print "length of program_vector :",len(program_vector)
		    	print "program_vector :",program_vector
		    	print "]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]"
			'''

                    except VectorizationFailedException:
                        program_vector = None

                    if program_vector is not None:
			print "debug..9.1"
                        input_.append(program_vector)
                    else:
			print "debug..9.2"
			print "vectorize failed" 
                        to_delete.append(i)
                        error_messages[idx].append('VectorizationFailed')

                # Delete
		print "debug..10"
                to_delete = sorted(to_delete)[::-1]

                for i in to_delete:
		    print "debug..11"
                    del entries[i]

                assert len(input_) == len(entries)

                if len(input_) == 0:
		    print "debug..12"
                    break

                # Pass it through the network
		print "debug..13"
                fix_vectors = get_fixes(network['session'], input_, network)
		print "fix_vectors =",fix_vectors
		print "debug..13.1"
                fixes = []

                # Devectorize them
                for i, fix_vector in enumerate(fix_vectors):
		    print "debug..14"
                    idx, _, _, _, _ = entries[i]
                    fix = devectorize(fix_vector, network['dictionary'])
		    print "fix =",fix
                    fixes_suggested_by_network[idx].append(fix)
                    fixes.append(fix)

                to_delete = []

                # Apply fixes
                for i, entry, fix in zip(range(len(fixes)), entries, fixes):
		    print "debug..15"
                    idx, program, name_dict, name_sequence, literal_sequence = entry

                    try:
                        program = sequences_of_programs[idx][-1]
                        program = apply_fix(program, fix, kind=fix_kind, check_literals=True)
			print "apply_fix passed after"
                        sequences_of_programs[idx].append(program)
			print "sequence_of_programs passed"
                        regen_source_code = tokens_to_source(program, name_dict, clang_format=True, literal_seq=literal_sequence)
			print "Oracle compile will"
                        this_errors, _ = compilation_errors(regen_source_code)
			print "Oracle compile was"
                        if len(fix.strip().split()) > 0 and len(this_errors) > len(errors[idx]):
                            to_delete.append(i)
			    print "ErrorIncreased!!!!!"
                            error_messages[idx].append('ErrorsIncreased')
                        else:
                            errors[idx] = this_errors
                    except IndexError:
                        to_delete.append(i)
                        error_messages[idx].append('IndexError')
                    except VectorizationFailedException as e:
                        to_delete.append(i)
                        error_messages[idx].append('VectorizationFailed')
                    except InvalidFixLocationException:
                        to_delete.append(i)

                        if fix.strip().split()[0] == '_eos_':
                            error_messages[idx].append('OK')
                        else:
                            error_messages[idx].append('InvalidFixLocation')
                    except SubstitutionFailedException:
                        to_delete.append(i)
                        error_messages[idx].append('SubstitutionFailed')
                    else:
			print "why else passed"
                        assert len(fix.strip().split()) == 0 or fix.strip().split()[0] != '_eos_'

                        if fix_kind == 'insert':
                            fix_ = ' '.join(fix.split()[1:])
                            fix_line = extract_line_number(fix_) + 1
                            fixes_to_return[idx].append('%s at line %d: %s' % (fix_kind, fix_line, ''.join(fix_to_source(fix_, program, name_dict, clang_format=True).split('\n'))))
                        else:
                            fix_line = extract_line_number(fix) + 1
                            fixes_to_return[idx].append('%s at line %d: %s' % (fix_kind, fix_line, ''.join(fix_to_source(fix, program, name_dict, name_seq=name_sequence, literal_seq=literal_sequence, clang_format=True).split('\n'))))

                # Delete
		print "debug..16"
                to_delete = sorted(to_delete)[::-1]

                for i in to_delete:
                    del entries[i]

        except KeyError as e:
            pass

        except InvalidFixLocationException:
            pass

        except SubstitutionFailedException:
            pass
        # -----------

        repaired_programs = {}

        for idx in sequences_of_programs:
            repaired_programs[idx] = tokens_to_source(sequences_of_programs[idx][-1], name_dict, clang_format=True, literal_seq=literal_sequence)
            repaired_programs[idx] = repaired_programs[idx].strip()

        return fixes_to_return, repaired_programs, error_messages
