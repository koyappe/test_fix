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

from util.helpers import get_lines, extract_line_number, recompose_program, FailedToGetLineNumberException, replace_ids
import regex as re

# Exceptions used by helpers
class SubstitutionFailedException(Exception):
    pass

class InvalidFixLocationException(Exception):
    pass

class VectorizationFailedException(Exception):
    pass

# Helper methods
def vectorize(tokens, vector_length, dictionary, normalize_names=False, reverse=False, append_eos=False):
    result = []
    print "token split length :",len(tokens.split())
    print "///////////////////"
    print "tokens :",tokens
    print "///////////////////"
    print "vector length :",vector_length
    if len(tokens.split()) > vector_length:
	print "token split length is longer than vector length"
        return None

    # Build up the list of scalars
    try:
        for token in tokens.split():
	    print "token :",token
            if normalize_names and '_<id>_' in token:
		print "add to dictionary '_<id>_1@'"
                result.append(dictionary[ '_<id>_1@'])
            else:
		print "add to dictionary others"
                result.append(dictionary[token])
    except KeyError as e:
	print "VectorizationFailedException"
        raise VectorizationFailedException(e.args[0])

    # Add EOS if requested
    if append_eos:
	print "add to dictionary '_eos_'"
        result.append(dictionary['_eos_'])

    # Compute padding
    padding_length = vector_length - len(result)
    padding = [dictionary['_pad_']] * padding_length

    # Pad as required
    if reverse:
	print "if reserve compute result"
        result = padding + result[::-1]
    else:
	print "if not reserve compute result"
        result = result + padding

    print "result :",result

    return result

def _get_reversed_dictionary(dictionary):
    result = {}

    for key, value in dictionary.iteritems():
        result[str(value)] = key

    return result

def devectorize(vector, dictionary, reverse=False):
    result = []
    reversed_dictionary = _get_reversed_dictionary(dictionary)

    for i, _ in enumerate(vector):
        if reverse:
            i = len(vector) - i - 1

        if reversed_dictionary[str(vector[i])] != '_pad_':
            result.append(reversed_dictionary[str(vector[i])])

    return ' '.join(result)

def apply_fix(program, fix, kind='replace', check_literals=False):
    print "apply_fix passed"
    # Break up program string into lines
    lines = get_lines(program)

    print "*******************"
    print "lines ="
    print lines
    print "*******************"
    print "lines length :",len(lines)
    # Truncate the fix
    fix = _truncate_fix(fix)
    print "*******************"
    print "fix ="
    print fix
    print "*******************"
    print "fix.split('~') :",fix.split('~')
    print "len(fix.split('~')) :",len(fix.split('~'))
    # Make sure there are two parts
    if len(fix.split('~')) != 2:
	print "InvalidFixLocationExeption"
	print "can not split 2 part"
        raise InvalidFixLocationException
    print "Retrieve insertion location"
    # Retrieve insertion location
    try:
	print "if replace 1"
        if kind == 'replace':
            fix_location = extract_line_number(fix)
	    print "kind == replace"
    	    print "*******************"
	    print "fix_location ="
	    print fix_location
    	    print "*******************"
        else:
            assert kind == 'insert'

            if fix.split()[0] != '_<insertion>_':
                print "Warning: First token did not suggest insertion (should not happen)"

            fix_location = extract_line_number(' '.join(fix.split()[1]))
    	    print "*******************"
	    print "fix_location =="
	    print fix_location
    	    print "*******************"
    except FailedToGetLineNumberException:
        raise InvalidFixLocationException
    print "Remove line number"
    # Remove line number
    fix = _remove_line_number(fix)

    print "*******************"
    print "fix ="
    print fix
    print "*******************"
    # Insert the fix
    if kind == 'replace':
	print "if replace 2" 
        try:
   	    check_literals = False #debug
            if lines[fix_location].count('_<id>_') != fix.count('_<id>_'):
		print "not include original id"
                raise SubstitutionFailedException
            if check_literals:
		print "check literals"
                for lit in ['string', 'char', 'number']:
                    if lines[fix_location].count('_<%s>' % lit) != fix.count('_<%s>_' % lit):
			print "not include original literal"
                        raise SubstitutionFailedException

            lines[fix_location] = replace_ids(fix, lines[fix_location])
        except IndexError:
	    print "InvalidFixLocationException"
            raise InvalidFixLocationException
    else:
        assert kind == 'insert'
        lines.insert(fix_location+1, fix)
    print "apply_fix end"
    return recompose_program(lines)

def _remove_line_number(fix):
    return fix.split('~')[1]

def _truncate_fix(fix):
    result = ''

    for token in fix.split():
        if token == '_eos_':
            break
        else:
            result += token + ' '

    return result.strip()

def _is_stop_signal(fix):
    if _truncate_fix(fix) == '':
        return True

def meets_criterion(incorrect_program_tokens, fix, name_dict, type_, name_seq=None, silent=True):
    lines = get_lines(incorrect_program_tokens)
    fix = _truncate_fix(fix)

    if _is_stop_signal(fix):
        #print 'is stop signal'
        return False

    try:
        fix_line_number = extract_line_number(fix)
    except Exception:
        #print 'failed to extract line number from fix'
        return False

    if fix_line_number >= len(lines):
        #print 'localization is pointing to line that doesn\'t exist'
        return False

    fix_line = lines[fix_line_number]

    # Make sure number of IDs is the same
    if len(re.findall('_<id>_\w*', fix_line)) != len(re.findall('_<id>_\w*', fix)):
        if not silent:
            print 'number of ids is not the same'
        return False

    keywords_regex = '_<keyword>_\w+|_<type>_\w+|_<APIcall>_\w+|_<include>_\w+'

    if type_ == 'replace' and re.findall(keywords_regex, fix_line) != re.findall(keywords_regex, fix):
        if not silent:
            print 'important words (keywords, etc.) change drastically'
        return False

    return True
