# cfgen

Uses a combination of Markov chains and context-free-grammars to generate random sentences with features of both language models.

*Created by William Gilpin, 2014-2016*


## Requirements and Installation

+ Python 2
+ [pyStatparser](https://github.com/bendavis78/pyStatParser)
+ NLTK
+ language-check (optional)

You can install the code and basic dependencies by running these commands

	$ git clone https://github.com/williamgilpin/cfgen
    $ conda install nltk
    $ pip install git+git://github.com/bendavis78/pyStatParser

For scoring grammar or automatically correcting the resulting sentences, install the Python Package [language-check](https://pypi.python.org/pypi/language-check).

    $ pip install 3to2
    $ pip install language-check

## Basic Usage

Point the tool to your corpus and set up the language model. Here we will use 2-grams of Mary Shelley's Frankenstein.

	mycorp = clean_corpus('cfgen/full_books/frankenstein.txt')
	tagged_corpus = tag_corpus(mycorp)
	termrules_mycorp = make_terminal_rules(tagged_corpus)
	my_kgram = make_kgram(mycorp, k=2)

Now generate an example sentence

	example_sentence = make_sentence(mycorp, termrules_mycorp, my_kgram)
    corrected_example_sentence = clean_output_text(example_sentence, use_language_tool=True)
    print(example_sentence + '\n')
    print(corrected_example_sentence)

A full workflow is given in the file **demos.ipynb**. 

<!-- I wrote about this project [on my blog.](https://gammacephei.wordpress.com/2014/08/17/algorithmic-trolling-of-social-networks/) -->

## Sample output

From a model trained on Mary Shelley's Frankenstein:

	these conversation possessed to his time
	those sense all fulfillment, me of affairs a soul been of I.
	The next kindness her black, light of it I wished as our perish certainty.
	horrible wretch that memory, their lake I to me, which the glad unhappiness 
	of the paroxysm of a several engagement of Clerval in her apartment and all 
	mainland, and on a progress, me surprised so divine me.


From a model trained on the Book of Revelations:

	Who said ever to no crown to was the listener and SECOND: 
	my man, and to the wheat,
	and had eat in this angel.

	Their pieces kill the sort the angel come up 
	to another translucent and weep any stone.
	Her timeless will measure them to the day, 
	hold created with earth noises and hurled every nation.

	There shown out upon the voice
	It be in seventh which is to trample, I.

	This tampering opened not for its time.

	The land to their moment Who threw their glory to cherish that art.

	The glory to the speaking, and at that white appearance, and say given 
	the thousand for the sake. And said show in myself. And it of no sweet victory 
	whose gateways enemies was loathe to the bowl
	and it for them and worked out as my hast to every vision.

	Their noise erase me.



<!-- ## TODO

+ Make the code automatically parse a subset of sentences in a corpus in order to generate a subsetted set of nonterminal grammar rules

+ Use Bayesian methods to randomly select among possible clause constructions based on previous clauses in the sentence, and use a Markov model to select words contextually based on higher level grammatical features of the sentence.

+ Punctuation is terrible right now because it has to be scraped off of hte corpus to prevent the tokenizer from choking.
 -->