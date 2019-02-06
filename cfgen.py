'''
A Python libraryfor generating random text using a context free grammar

Future: pyStatParser needs to be replaced. Currently quotes are excluded from
the training corpus because the parser does not do a very good job with them

William Gilpin, 2014-2019
'''

import nltk
from nltk import data, CFG, ChartParser, Nonterminal
from nltk.tokenize import word_tokenize
from random import choice

from numpy import median, floor
from numpy.random import rand

# import ATD
import time

import string
import time
import glob
import random
import os.path
import warnings

try:
    from stat_parser import Parser
    has_parser = True
except ImportError:
    warnings.warn('Could not import pyStatParser. Sentence structures will be' \
                  ' randomly selected from a pre-computed set of rules instead.')
    has_parser = False
    
try: 
    import language_check
    has_language_tool = True
except ImportError:
    warnings.warn('Could not import language-check or LanguageTool. Rule-based' \
                   ' post-processing will not be applied to generated text.')
    has_language_tool = False
    

# GLOBAL: characters that need to be replaced after pos tagging is complete    
to_replace = [',','.',':',';',"''",'"','``','(',')','$','+','\'','-LRB-','-RRB-']
replacements = ['xcomma','xperiod','xcolon','xsemicolon','xquote','xquote', \
                'xquote','openparen','closeparen','xdollar','xplus','xapostrophe','openparen','closeparen']
#excluded_items = ['\t','\n','\r','-\n','\"','|','--']
excluded_items1 = ['\t','\n','\r','-\n','|','--'] # replaced with space
excluded_items2 = ['\"','\''] # replaced with nothing
rep_pairs = [('. ,',','),
        (' i ',' I '),
        (', ,',','),
        ('. .','.'),
        (' . ','. '),
        (' , ',', '),
        (' : ',': '),
        (' ; ','; ')]

def is_terminal(symb):
    '''determine if a symbol is terminal
    
    Parameters
    ----------
    
    symb : str
    
    Returns
    -------
    
    out : bool
        whether if symb is terminal

    '''
    out = hasattr(symb, '__hash__') and not isinstance(symb, Nonterminal)
    return out

def list_overlaps(list1, list2, asymmetric=False):
    '''
    Find overlapping elements in a list, including repeats
    if asymmetric=True, repeats will only matter for second argument

    Parameters
    ----------

    list1 : list
    list2 : list
        The two lists that are being compared

    asymmetric : bool
        count only redundancies in the second list
        towards the total


    '''
    if asymmetric:
        member_set = set(list1)
    else:
        member_set = list1

    all_overlaps = list()
    for member1 in member_set:
        commons = [member2 for member2 in list2 if member2==member1]
        all_overlaps.extend(commons)

    return all_overlaps
    



class GrammarModel(object):
    """
    Parses a corpus for use in a context-free grammar model
    
    Args:
        corpus_path : string denoting a .txt file to parse
        model_order : the order of the kgram table to use
        clean_text : a boolen denoting whether to use a spellchecker to 
            clean up the generated text (only works if languagetool installed)
    """
    def __init__(self, corpus_path, model_order, clean_text=False):
        self.order = model_order
        self.corpus = self.clean_corpus(corpus_path)
        self.kgram_table = self.make_kgram_table()
        
        self.tags = self.tag_corpus()
        self.term_rules = self.make_terminal_rules()
        
        self.do_clean = clean_text
        if self.do_clean and not has_language_tool:
            warnings.warn("LanguageTool not installed, cannot fix grammar errors automatically")
    
    def make_kgram_table(self, clean=True):
        '''
        clean : bool
            If activated, remove some troublesome characters
            (introduces risk of markov chain throwing a KeyError)
        add functions to strip all capitals and punctuation from a corpus
        (only run punctuation one internally here)
        '''
        k = self.order
        clean_corpus = self.corpus 
        if clean:
            swappairs = zip(to_replace, replacements)
            for member in swappairs:
                clean_corpus = clean_corpus.replace(member[0],member[1])


        mywords = word_tokenize(clean_corpus)
        kgrams = dict()
        for ii in range(len(mywords)-k):
            keyname = ' '.join(mywords[ii:ii+k])
            if keyname in kgrams.keys():
                kgrams[keyname].append(mywords[ii+k])
            else: 
                kgrams[keyname] = [mywords[ii+k]]
        return kgrams
    
    def clean_corpus(self, path, make_lowercase=True):
        '''
        Clean up a text corpus by removing characters and expressions that 
        do not matter at the level of individual sentences

        path : str
            A string to a .txt file containing a corpus

        make_lowercase : bool
            Convert corpus to all lowercase

        '''
        filename_root = os.path.dirname(path)
        corpus_members = glob.glob(path)
        corpus = ''

        # get rid of random line breaks and exclude troublesome expressions like quotes
        for member in corpus_members:
            with open (member, "r") as openfile:
                data = openfile.read()
                data = data.replace('.','.')
                data = data.replace('\'\'','"')
                data = data.replace('``','"')
                data = data.replace('`','\'')
                data = data.replace(',',',')
                data = data.replace(';',';')
                data = data.replace(':',':')
                for badchar in excluded_items1:
                    data = data.replace(badchar, ' ')
                for badchar in excluded_items2:
                    data = data.replace(badchar, '')
            corpus = corpus + ' ' + data
        if make_lowercase:
            corpus = corpus.lower()

        return corpus

   
    def tag_corpus(self):
        '''
        Use NLTK to identify the linguistic function of
        the words in a corpus. 

        This is only necessary to compile the full list of possible
        terminal symbols. And so Terminal tokens that require special rules
        (like punctuation) are excluded. This method can likely be combined
        with the other initialization functions into a subclass
        
        Args:
            corpus : str
                A corpus that has been stripped of all troublesome
                characters using the clean_corpus() function
        Returns:
            revised_tokens : list of tuples
                A list of tuples consisting of a word in position 1,
                and its function within the sentence in position 2
        '''

        tokens = nltk.word_tokenize(self.corpus)
        pos_tagged_tokens = nltk.pos_tag(tokens)

        swappairs = zip(to_replace, replacements)

        clean_tags = list()
        for tupe in pos_tagged_tokens:
            fixed_tupe = tupe + tuple()  
            for member in swappairs:
                fixed_tupe = tuple([item.replace(member[0], member[1]) for item in fixed_tupe])
            clean_tags.append(fixed_tupe)

        return clean_tags
    
    def make_terminal_rules(self):
        '''
        Searches through a list of tagged words and obtain
        all of the Terminal characters

        TODO: Clean up the troublesome terminal characters

        Arg:
            path : str

        '''

        all_rules = ''

        tags = list({tupe[1] for tupe in self.tags})

        badtags = ['#','$',',','-NONE-','.',':','TO',"''",'(',')'] + replacements
        # badtags = ['#','$',',','-NONE-','.',':','TO','POS',"''",'(',')'] # bad terminal tags
        tags = [item for item in tags if item not in badtags]
        for tag in tags:
            allsyms = [('\'' + tupe[0] + '\'') for tupe in self.tags if tupe[1]==tag]
            gr_rule = (tag + " -> ")
            gr_rule += ' | '.join(allsyms)
            gr_rule += '\n'
            gr_rule = gr_rule.replace('PRP$','PRPx')
            gr_rule = gr_rule.replace('WP$','WPx')
            gr_rule = gr_rule.replace('-LRB-','xLRBx')
            gr_rule = gr_rule.replace('-RRB-','xRRBx')
            all_rules += gr_rule

        #all_rules +=('''xapostrophe -> "\'"\n''')
        all_rules +=('''TO -> 'to'\n''')
        all_rules +=('''xLRBx -> 'openparen'\n''')
        all_rules +=('''xRRBx -> 'closeparen'\n''')
        all_rules +=('''POS -> 'xapostrophes'\n''')

        for item in replacements:
            rep_rule = item + ' -> \'' + item + '\'\n'
            all_rules += rep_rule

        return all_rules


    def parse_sentence(self, sentence_str):
        '''
        Generate nonterminal rules using a stochastic sentence parser

        Args:
            my_sentence : str
                A single sentence (str) 

        '''

        parser = Parser()
        parsee = parser.parse(sentence_str)

        rules = ""

        # possibly add: brackets, double quotes

        for production in parsee.productions():
            if not self.is_terminal(production.rhs()[0]):
                rules += str(production) + '\n'

        # now re-tag special characters
        swappairs = zip(to_replace, replacements)
        for member in swappairs:
            rules = rules.replace(member[0],member[1])

        return rules


    def make_sentence_markov(self, nwords, start_word=''):
        '''
        Use a k-word markov model to generate text
        
        Args: 
            nwords : The number of words of output text to generate
            start_word : the first word of the sentence, can be left
                empty.
        
        Returns:
            out : string corresponding to the generated sentence
        
        '''
        all_keys = list(self.kgram_table.keys())
        mykval = len(all_keys[0].split(' '))
        
        if not start_word:
            start_word = choice(all_keys)
        
        sent = start_word.split(' ')
        for ii in range(nwords):
            prevkey = ' '.join(sent[-mykval:])
            new_txt = choice(self.kgram_table[prevkey])
            sent.append(new_txt)
        out = ' '.join(sent)
        
        # add a post-processing step here
        out = out.replace('xcomma', ',')
        out = out.replace('xperiod', '.')
        out = out.replace('xsemicolon', ';')
        
        return out

    def make_random_sentence(nwords):
        '''
        Make a random sentence of a given length from the corpus
        Used for testing as a null model
        
        nwords : int
        '''

        clean_corpus = self.corpus 
        
        #if clean:
        #    swappairs = zip(to_replace, replacements)
        #    for member in swappairs:
        #        clean_corpus = clean_corpus.replace(member[0], member[1])

        mywords = word_tokenize(self.corpus)
        sent = choice(mywords, nwords)
        out = ' '.join(sent)
        # add a post-processing step here
        out = out.replace('xcomma', ',')
        out = out.replace('xperiod', '.')
        out = out.replace('xsemicolon', ';')
        
        return out
    
    def is_terminal(self, symb):
        '''
        Determines if a symbol is terminal

        Args:
            symb : str
        Returns:
            out : bool
                whether if symb is terminal
        '''
        out = hasattr(symb, '__hash__') and not isinstance(symb, Nonterminal)
        return out


    def produce(self, grammar, symbol, depth=0, maxdepth=25):
        '''
        Generate the next word of the sentence randomly, without
        using the Markov chain

        Args:
            grammar : nltk.grammar.CFG
            symbol : nltk.grammar.Nonterminal
            depth : int
                The depth of the recursive tree search
            maxdepth : int
                The maximum allowed recursion depth before throwing a
                ValueError

        TODO: make a custom UserError type

        '''
        if depth > maxdepth:
            raise ValueError('Recursion went too deep, one of the example syntax' \
                             ' sentences might be poorly formed or poorly parsed')
        words = []

        productions = grammar.productions(lhs = symbol)
        production = choice(productions)

        for sym in production.rhs():
            if self.is_terminal(sym):
                words.append(sym)
            else:
                words.extend(self.produce(grammar, sym, depth=(depth+1), maxdepth=maxdepth))
        return words



    def produce_kgram(self, grammar, symbol, depth=0, maxdepth=25, sent=[]):
        '''
        Args:
            grammar : nltk.grammar.CFG
            symbol : nltk.grammar.Nonterminal
            kgram_dict : dict
                A dictionary with k-words as keys and all
                of the following words as vals
            depth : int
                The depth of the recursive tree search
            maxdepth : int
                The maximum allowed recursion depth before throwing a
                ValueError
            sent : list of str
                The entirety of the sentence so far
            TODO: make a custom UserError type
        ''' 
        kgram_dict = self.kgram_table
        k = len( list(kgram_dict.keys())[0].split(' ') )

        if depth > maxdepth:
            raise ValueError('Recursion went too deep, one of the example syntax' \
                             ' sentences might be poorly formed or poorly parsed')
        words = []

        productions = grammar.productions(lhs = symbol)
        productions_clone = list(productions)

        if len(sent) >= k:
            key_val = str(' '.join(sent[-k:]))
        else:
            key_val = ''


        if (len(productions_clone) > 10) and (len(sent) >= k) and  key_val in kgram_dict.keys():
            all_production_words = [str(item.rhs()[0]) for item in productions_clone]
            candidate_words = kgram_dict[key_val]
            valid_words = list_overlaps(all_production_words, candidate_words, asymmetric=True)
            if len(valid_words) > 0:
                production_word = choice(valid_words)
                production = list(grammar.productions(rhs = production_word))[0]
            else:
                production = choice(productions)
        else:
            production = choice(productions)

        for sym in production.rhs():
            if self.is_terminal(sym):
                words.append(sym)
                sent.append(sym)
            else:
                words.extend(self.produce_kgram(grammar, sym, depth=(depth+1), maxdepth=maxdepth, sent=sent))

        return words


    def make_sentence(self, do_markov=True, **kwargs):
        '''

        Generate sentences with random structure and word choice
        using a context-free grammar

        The start point is taken from the sentence itself.

        Parameters
        ----------
    
        do_markov : boolean that can be used to toggle the Markov
            word selection on or off
        
        maxdepth : int
            The maximum allowed recursion depth before throwing a
            ValueError

        fixed_grammar : bool
            Turn off the random sentence selection and used a fixed grammar
            instead.

        sample_sentence : str
            When fixed_grammar is turned on, this is the sentence that will
            be parsed. This can be finicky with grammars containing specially
            punctuated constructions like quotations or positions


        Notes
        -----

        Add the ability to turn off the kgram parsing, ideally by counting
        the number of unnamed arguments
        ----> Added this option

        '''
        
        corpus = self.corpus
        term_rules = self.term_rules
        
        if (self.order > 0) and do_markov:
            markov_flag = True
        else:
            markov_flag = False

        fixed_grammar = kwargs.pop('fixed_grammar', False)
        sample_sentence = kwargs.pop('sample_sentence', '')
        maxdepth = kwargs.pop('maxdepth', 25)

        if fixed_grammar:
            if sample_sentence=='':
                warnings.warn('When using fixed_grammar, user should specify ' \
                              'the keyword argument "sample_sentence." Using a default simple sentence.')
                sample_sentence = 'The cow jumped over the moon.'
            else:
                pass


        flag = False
        attempts = 0
        while not flag and attempts < 30:
            tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

            if has_parser and not fixed_grammar:  
                rsent = choice(tokenizer.tokenize(corpus))
            elif fixed_grammar:
                rsent = sample_sentence
            elif not has_parser and not fixed_grammar:
                # select from a parsed corpus of pre-approved grammars
                warnings.warn("Usage library being built, falling back to simple sentence")
                rsent = "The dog walked up the stairs slowly."
            else:
                warnings.warn("Usage library being built, falling back to simple sentence")
                rsent = "The dog walked up the stairs slowly."

            parsed_syntax = self.parse_sentence(rsent)
            # print(parsed_syntax)
            cfg_str = term_rules + parsed_syntax
            try:  
                startpt = parsed_syntax[:parsed_syntax.find(' ->')]
                startpt = nltk.grammar.Nonterminal(startpt)
                grammar = CFG.fromstring(cfg_str)
                parser = ChartParser(grammar)
                gr = parser.grammar()
                if markov_flag:
                    out_txt = (' '.join(self.produce_kgram(gr, startpt, maxdepth=maxdepth, sent=[])) )
                else:
                    out_txt = (' '.join(self.produce(gr, startpt, maxdepth=maxdepth)) )
                flag = True
            except ValueError:
                warnings.warn('Badly formed sentence encountered, resampling the corpus.')
                attempts += 1

        # now re-tag special characters
        swappairs = zip(replacements,to_replace)
        for member in swappairs:
            out_txt = out_txt.replace(member[0],member[1])


        return out_txt    



###############-------------------------#####################
# functions for processing text output


def clean_output_text(output_text, use_language_tool=False):
    '''
    Post-processing to clean up the output returned by the
    text generation program
    
    This uses the rule-based grammar checking of Language Tool
    to correct minor capitalization and tense issues in the 
    outputted text
    
    Parameters
    ----------
    
    output_text : str
    
    use_language_tool : bool
        Whether to use LanguageTool to automatically clean up
        the output text
    
    '''
    
    swappairs = zip(replacements,to_replace)
    for member in swappairs:
        output_text = output_text.replace(member[0], member[1])
    
    for member in rep_pairs:
        output_text = output_text.replace(member[0], member[1])
    
    if has_language_tool and use_language_tool:
        tool = language_check.LanguageTool('en-US')
        matches = tool.check(output_text)
        output_text = language_check.correct(output_text, matches)
        output_text = str(output_text)
    
    return output_text


def sample_corpus(corpus, samp_len):
    '''Take a random continuous set of words from a text, 
    return the sample and the text with the passage excised.
    
    Parameters
    ----------
    corpus : str
    
    samp_len : int
    
    Returns
    -------
    
    text_sample, reduced_corpus : str, str
    
    '''
    
    all_words = corpus.split(' ') 
    nrange = len(all_words) - samp_len - 1
    nwords = len(all_words)
    
    sample_index = choice(range(nrange))
    
    text_sample = ' '.join(all_words[sample_index:sample_index+samp_len])
    reduced_corpus = ' '.join(all_words[:sample_index] + all_words[sample_index+samp_len:])
    
    return text_sample, reduced_corpus

def all_common_substring(s1, s2,threshold_length=15):
    '''
    Return a list of all substrings of a given length that two
    strings have in common
    
    Based on standard code for solving the "longest common substring" problem
    
    '''
    matches = [[0]*(1 + len(s2)) for i in range(1 + len(s1))]
    all_sub = list()
    longest, x_longest = 0, 0
    for x in range(1, 1+len(s1)):
        for y in range(1, 1+len(s2)):
            if s1[x-1] == s2[y-1]:
                matches[x][y] = matches[x-1][y-1] + 1
                if matches[x][y] == threshold_length:
                    longest = matches[x][y]
                    x_longest = x
                    sout = s1[x_longest - longest: x_longest]
                    all_sub.append(sout)
            else:
                matches[x][y] = 0
    return all_sub


def similarity_score(s1, s2, threshold_length='auto'):
    '''
    Compute the similarity between two strings based on the
    number of identical substrings of at least a given length
    
    Args:
    s1, s2 : The two strings to compare
    threshold_length : int
        The length for overlapping substrings to be significant
        If this is not specified, it is set to thrice the median
        length of words in the two strings
        
    Returns:
        score : the similarity score, between 0.0 and 1.0
    
    '''
    if threshold_length=='auto':
        ave_word_len = median([len(item) for item in (s1 + ' ' + s2).split(' ')])
        threshold_length = int(3*ave_word_len)
    
    min_len = max([len(s1), len(s2)])
    max_sim = floor(min_len/float(threshold_length))
    
    all_comm = all_common_substring(s1, s2, threshold_length=threshold_length)
    
    score = float(len(all_comm))/max_sim
    
    return score


def rand_str(str_len):
    '''Returns a random string of the specified length'''
    random_string = ''
    char_set = 'abcdefghijklmnopqrstuvwxyz'
    for ii in range(str_len):
        letter_ind = int(floor(rand()*len(char_set)))
        random_string += char_set[letter_ind]   
    return random_string

def grammar_score_raw(some_text):
    '''
    Find all grammatical errors in a text
    
    Excludes cosmetic errors, like misuse of capitals, 
    and instead focus on structural issues
    '''
    tool = language_check.LanguageTool('en-US')
    matches = tool.check(some_text)

    structural_errors = list()
    for item in matches:
        if item.ruleId.find('WHITESPACE') != -1:
            continue
        elif item.ruleId.find('UPPERCASE') != -1:
            continue
        elif item.ruleId.find('LOWERCASE') != -1:
            continue
        elif item.ruleId.find('MORFOLOGIK_RULE_EN_US') != -1:
            continue
        elif item.ruleId.find('ENGLISH_WORD_REPEAT_BEGINNING_RULE') != -1:
            continue
        else:
            
            error_words = some_text[item.fromx:item.tox]
            
            replacement_word = item.replacements
            
            err_ind = item.contextoffset

            try:
                prior_word = some_text[:err_ind].split(' ')[-2]
            except IndexError:
                prior_word = ''
            
            structural_errors.append((error_words, prior_word, replacement_word))
    
    return structural_errors

### Functions for analyzing output text ###

def grammar_score_atd_raw(some_text):
    
    ATD.setDefaultKey("cfgen call " + rand_str(5))

    wait_time = 1
    have_answer = False

    while not have_answer:
        try:
            error_list = ATD.checkDocument(some_text)
            have_answer = True
        except:
            print('ATD API is having trouble!')
            print('waiting for: ' + str(wait_time) + ' sec.')
            time.sleep(wait_time)
            ATD.setDefaultKey("cfgen call " + rand_str(5))
            wait_time = wait_time + 5
    
    all_errors = []
    for error in error_list:
        if error.type!='grammar':
            continue
        else:
            all_errors.append((error.string, error.precontext))
    return all_errors

def grammar_score_comparative(some_text):
    '''
    Detects similar errors based just on whether the
    affected word is the same in both grammar checking tools
    '''
    
    results_langtool = grammar_score_raw(some_text)
    results_atd = grammar_score_atd_raw(some_text)
    
    err_words_langtool = [item[1] for item in results_langtool]
    err_words_atd = [item[1] for item in results_atd]
    
    common_errs = set(err_words_atd).intersection(err_words_langtool)
    
    nboth = float(len(common_errs))
    n1 = float(len(err_words_atd))
    n2 = float(len(err_words_langtool))
    nmiss = (n1 - nboth)*(n2 - nboth)/nboth
    
    return((n1,n2,nmiss))
    
def grammar_score(some_text, method='langtool'):
    '''
    Counts the total number of errors in a text
    Excludes cosmetic errors, like misuse of capitals, 
    and instead focus on structural issues
    
    Args:
        some_text : str, the text to be checked
        method : str, either 'langtool' or 'atd'
    '''
    if method=='langtool':
        structural_errors = grammar_score_raw(some_text)
    elif method=='atd':
        structural_errors = grammar_score_atd_raw(some_text)
    else:
        warnings.ward("Unknown method choice for grammar score")
        structural_errors = []
    
    error_score = float(len(structural_errors))/len(some_text)
    
    return error_score

def score_sentences(fname, delim='. ',wait_time=0):
    '''

    Args:
        fname : str
            The path to a txt file containing a bunch of sentences split by some
            delimiter
        delim : str
            The delimiter between sentences. default period followed by space
        wait_time : float
            The time (in seconds) to wait between successive calls to the ATD
            API. If a strange error is being returned, try increasing this time      
    Returns: 
        scores : 2xN np.array
            An array of scores for each sentence  
    Examples:
        score_sentences('my_sentences.txt', delim='\n',wait_time=.25)
    '''
    with open(fname) as txt_file:
        sents = txt_file.read()
        
    sents = sents.split(delim)
    sents = [thing for thing in sents if thing not in ['', ' ','.']]

    all_grammar_scores_langtool = []
    all_grammar_scores_atd = []

    #ii=0
    for sent in sents:
        #print(sent)
        all_grammar_scores_langtool.append( grammar_score(sent,method='langtool') )
        all_grammar_scores_atd.append( grammar_score(sent,method='atd') )
        #print(ii);ii=ii+1
        time.sleep(wait_time)
        
    scores = array([all_grammar_scores_langtool,all_grammar_scores_atd])
    scores = scores.T
    
    return(scores)