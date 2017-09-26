
.. _textmining:

===========
Text Mining
===========

.. note::

   Sharpening the knife longer can make it easier to hack the firewood -- old Chinese proverb

.. figure:: images/sen_word_freq.png
   :align: center


Text Collection 
+++++++++++++++


Image to text
-------------

.. code-block:: python

	def img2txt(img_dir):
	    """
	    convert images to text
	    """
	    import os, PythonMagick
	    from datetime import datetime
	    import PyPDF2

	    from PIL import Image
	    import pytesseract

	    f = open('doc4img.txt','wa')
	    for img in [img_file for img_file in os.listdir(img_dir)
	                if (img_file.endswith(".png") or 
	                    img_file.endswith(".jpg") or 
	                    img_file.endswith(".jpeg"))]:

	        start_time = datetime.now()

	        input_img = img_dir + "/" + img

	        print('--------------------------------------------------------------------')
	        print(img)
	        print('Converting ' + img +'.......')
	        print('--------------------------------------------------------------------')     

	        # extract the text information from images
	        text = pytesseract.image_to_string(Image.open(input_img))
	        print(text)
	        
	        # ouput text file 
	        f.write( img + "\n")
	        f.write(text.encode('utf-8'))
	        

	        print "CPU Time for converting" + img +":"+ str(datetime.now() - start_time) +"\n"
	        f.write( "\n-------------------------------------------------------------\n")

	    f.close()   

Image Enhnaced to text
----------------------

.. code-block:: python

	def pdf2txt_enhance(img_dir,scaler):
	    """
	    convert images files to text
	    """
	    
	    import numpy as np
	    import os, PythonMagick
	    from datetime import datetime
	    import PyPDF2

	    from PIL import Image, ImageEnhance, ImageFilter
	    import pytesseract

	    f = open('doc4img.txt','wa')
	    for img in [img_file for img_file in os.listdir(img_dir)
	                if (img_file.endswith(".png") or 
	                    img_file.endswith(".jpg") or 
	                    img_file.endswith(".jpeg"))]:

	        start_time = datetime.now()

	        input_img = img_dir + "/" + img
	        enhanced_img = img_dir + "/" +"Enhanced" + "/"+ img
	        
	        im = Image.open(input_img) # the second one
	        im = im.filter(ImageFilter.MedianFilter())
	        enhancer = ImageEnhance.Contrast(im)
	        im = enhancer.enhance(1)
	        im = im.convert('1')
	        im.save(enhanced_img)
	        
	        for scale in np.ones(scaler):
	            im = Image.open(enhanced_img) # the second one 
	            im = im.filter(ImageFilter.MedianFilter())
	            enhancer = ImageEnhance.Contrast(im)
	            im = enhancer.enhance(scale)
	            im = im.convert('1')
	            im.save(enhanced_img)
	        


	        print('--------------------------------------------------------------------')
	        print(img)
	        print('Converting ' + img +'.......')
	        print('--------------------------------------------------------------------')     

	        # extract the text information from images
	        text = pytesseract.image_to_string(Image.open(enhanced_img))
	        print(text)
	        
	        # ouput text file 
	        f.write( img + "\n")
	        f.write(text.encode('utf-8'))
	        

	        print "CPU Time for converting" + img +":"+ str(datetime.now() - start_time) +"\n"
	        f.write( "\n-------------------------------------------------------------\n")

	    f.close()   

PDF to text
-----------

.. code-block:: python

	def pdf2txt(pdf_dir,image_dir):
	    """
	    convert PDF to text
	    """
	    
	    import os, PythonMagick
	    from datetime import datetime
	    import PyPDF2

	    from PIL import Image
	    import pytesseract

	    f = open('doc.txt','wa')
	    for pdf in [pdf_file for pdf_file in os.listdir(pdf_dir) if pdf_file.endswith(".pdf")]:

	        start_time = datetime.now()

	        input_pdf = pdf_dir + "/" + pdf

	        pdf_im = PyPDF2.PdfFileReader(file(input_pdf, "rb"))
	        npage = pdf_im.getNumPages()

	        print('--------------------------------------------------------------------')
	        print(pdf)
	        print('Converting %d pages.' % npage)
	        print('--------------------------------------------------------------------')     

	        f.write( "\n--------------------------------------------------------------------\n")
	        
	        for p in range(npage):

	            pdf_file = input_pdf + '[' + str(p) +']'
	            image_file =  image_dir  + "/" + pdf+ '_' + str(p)+ '.png'

	            # convert PDF files to Images
	            im = PythonMagick.Image()
	            im.density('300')
	            im.read(pdf_file)
	            im.write(image_file)

	            # extract the text information from images
	            text = pytesseract.image_to_string(Image.open(image_file))

	            #print(text)

	            # ouput text file 
	            f.write( pdf + "\n")
	            f.write(text.encode('utf-8'))
	        

	        print "CPU Time for converting" + pdf +":"+ str(datetime.now() - start_time) +"\n"

	    f.close()   
    

Text Preprocessing 
++++++++++++++++++

* check to see if a row only contains whitespace

.. code-block:: python

	def check_blanks(data_str):
	    is_blank = str(data_str.isspace())
	    return is_blank


* Determine whether the language of the text content is english or not: Use langid module to classify the language to make sure we are applying the correct cleanup actions for English langid

.. code-block:: python

	def check_lang(data_str):
	    predict_lang = langid.classify(data_str)
	    if predict_lang[1] >= .9:
	        language = predict_lang[0]
	    else:
	        language = 'NA'
	    return language

* Remove features

.. code-block:: python

	def remove_features(data_str):
	    # compile regex
	    url_re = re.compile('https?://(www.)?\w+\.\w+(/\w+)*/?')
	    punc_re = re.compile('[%s]' % re.escape(string.punctuation))
	    num_re = re.compile('(\\d+)')
	    mention_re = re.compile('@(\w+)')
	    alpha_num_re = re.compile("^[a-z0-9_.]+$")
	    # convert to lowercase
	    data_str = data_str.lower()
	    # remove hyperlinks
	    data_str = url_re.sub(' ', data_str)
	    # remove @mentions
	    data_str = mention_re.sub(' ', data_str)
	    # remove puncuation
	    data_str = punc_re.sub(' ', data_str)
	    # remove numeric 'words'
	    data_str = num_re.sub(' ', data_str)
	    # remove non a-z 0-9 characters and words shorter than 3 characters
	    list_pos = 0
	    cleaned_str = ''
	    for word in data_str.split():
	        if list_pos == 0:
	            if alpha_num_re.match(word) and len(word) > 2:
	                cleaned_str = word
	            else:
	                cleaned_str = ' '
	        else:
	            if alpha_num_re.match(word) and len(word) > 2:
	                cleaned_str = cleaned_str + ' ' + word
	            else:
	                cleaned_str += ' '
	        list_pos += 1
	    return cleaned_str

* removes stop words

.. code-block:: python

	def remove_stops(data_str):
	    # expects a string
	    stops = set(stopwords.words("english"))
	    list_pos = 0
	    cleaned_str = ''
	    text = data_str.split()
	    for word in text:
	        if word not in stops:
	            # rebuild cleaned_str
	            if list_pos == 0:
	                cleaned_str = word
	            else:
	                cleaned_str = cleaned_str + ' ' + word
	            list_pos += 1
	    return cleaned_str

* tagging text

.. code-block:: python

	def tag_and_remove(data_str):
	    cleaned_str = ' '
	    # noun tags
	    nn_tags = ['NN', 'NNP', 'NNP', 'NNPS', 'NNS']
	    # adjectives
	    jj_tags = ['JJ', 'JJR', 'JJS']
	    # verbs
	    vb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
	    nltk_tags = nn_tags + jj_tags + vb_tags

	    # break string into 'words'
	    text = data_str.split()

	    # tag the text and keep only those with the right tags
	    tagged_text = pos_tag(text)
	    for tagged_word in tagged_text:
	        if tagged_word[1] in nltk_tags:
	            cleaned_str += tagged_word[0] + ' '

	    return cleaned_str

* lemmatization

.. code-block:: python	   

	def lemmatize(data_str):
	    # expects a string
	    list_pos = 0
	    cleaned_str = ''
	    lmtzr = WordNetLemmatizer()
	    text = data_str.split()
	    tagged_words = pos_tag(text)
	    for word in tagged_words:
	        if 'v' in word[1].lower():
	            lemma = lmtzr.lemmatize(word[0], pos='v')
	        else:
	            lemma = lmtzr.lemmatize(word[0], pos='n')
	        if list_pos == 0:
	            cleaned_str = lemma
	        else:
	            cleaned_str = cleaned_str + ' ' + lemma
	        list_pos += 1
	    return cleaned_str



**define the preprocessing function in PySpark**

.. code-block:: python

	from pyspark.sql.functions import udf
	from pyspark.sql.types import StringType
	import preproc as pp

	check_lang_udf = udf(pp.check_lang, StringType())
	remove_stops_udf = udf(pp.remove_stops, StringType())
	remove_features_udf = udf(pp.remove_features, StringType())
	tag_and_remove_udf = udf(pp.tag_and_remove, StringType())
	lemmatize_udf = udf(pp.lemmatize, StringType())
	check_blanks_udf = udf(pp.check_blanks, StringType())


Text Classification 
+++++++++++++++++++

.. code-block:: python

	from nltk.stem.wordnet import WordNetLemmatizer
	from nltk.corpus import stopwords
	from nltk import pos_tag
	import string
	import re
	import langid

Sentiment analysis
++++++++++++++++++

Introduction
------------

`Sentiment analysis`_ (sometimes known as opinion mining or emotion AI) refers to the use of natural language processing, text analysis, computational linguistics, and biometrics to systematically identify, extract, quantify, and study affective states and subjective information. Sentiment analysis is widely applied to voice of the customer materials such as reviews and survey responses, online and social media, and healthcare materials for applications that range from marketing to customer service to clinical medicine.

Generally speaking, sentiment analysis aims to **determine the attitude** of a speaker, writer, or other subject with respect to some topic or the overall contextual polarity or emotional reaction to a document, interaction, or event. The attitude may be a judgment or evaluation (see appraisal theory), affective state (that is to say, the emotional state of the author or speaker), or the intended emotional communication (that is to say, the emotional effect intended by the author or interlocutor).

Sentiment analysis in business, also known as opinion mining is a process of identifying and cataloging a piece of text according to the tone conveyed by it. It has broad application:

* Sentiment Analysis in Business Intelligence Build up
* Sentiment Analysis in Business for Competitive Advantage
* Enhancing the Customer Experience through Sentiment Analysis in Business

Pipeline
--------

.. _fig_sa_pipeline:
.. figure:: images/sentiment_analysis_pipeline.png
   :align: center

   Sentiment Analysis Pipeline

Demo
----

1. Set up spark context and SparkSession

.. code-block:: python

	from pyspark.sql import SparkSession

	spark = SparkSession \
	    .builder \
	    .appName("Python Spark Sentiment Analysis example") \
	    .config("spark.some.config.option", "some-value") \
	    .getOrCreate()

2. Load dataset

.. code-block:: python

	df = spark.read.format('com.databricks.spark.csv').\
	                               options(header='true', \
	                               inferschema='true').\
	            load("../data/newtwitter.csv",header=True);

.. code-block:: python

	+--------------------+----------+-------+
	|                text|        id|pubdate|
	+--------------------+----------+-------+
	|10 Things Missing...|2602860537|  18536|
	|RT @_NATURALBWINN...|2602850443|  18536|
	|RT @HBO24 yo the ...|2602761852|  18535|
	|Aaaaaaaand I have...|2602738438|  18535|
	|can I please have...|2602684185|  18535|
	+--------------------+----------+-------+
	only showing top 5 rows

3. Text Preprocessing

* remove non ASCII characters

.. code-block:: python

	from pyspark.sql.functions import udf
	from pyspark.sql.types import StringType

	from nltk.stem.wordnet import WordNetLemmatizer
	from nltk.corpus import stopwords
	from nltk import pos_tag
	import string
	import re

	# remove non ASCII characters
	def strip_non_ascii(data_str):
	    ''' Returns the string without non ASCII characters'''
	    stripped = (c for c in data_str if 0 < ord(c) < 127)
	    return ''.join(stripped)
	# setup pyspark udf function    
	strip_non_ascii_udf = udf(strip_non_ascii, StringType()) 

check:
.. code-block:: python

	df = df.withColumn('text_non_asci',strip_non_ascii_udf(df['text']))
	df.show(5,True)

ouput:

.. code-block:: python

	+--------------------+----------+-------+--------------------+
	|                text|        id|pubdate|       text_non_asci|
	+--------------------+----------+-------+--------------------+
	|10 Things Missing...|2602860537|  18536|10 Things Missing...|
	|RT @_NATURALBWINN...|2602850443|  18536|RT @_NATURALBWINN...|
	|RT @HBO24 yo the ...|2602761852|  18535|RT @HBO24 yo the ...|
	|Aaaaaaaand I have...|2602738438|  18535|Aaaaaaaand I have...|
	|can I please have...|2602684185|  18535|can I please have...|
	+--------------------+----------+-------+--------------------+
	only showing top 5 rows


* fixed abbreviation

.. code-block:: python

	# fixed abbreviation
	def fix_abbreviation(data_str):
	    data_str = data_str.lower()
	    data_str = re.sub(r'\bthats\b', 'that is', data_str)
	    data_str = re.sub(r'\bive\b', 'i have', data_str)
	    data_str = re.sub(r'\bim\b', 'i am', data_str)
	    data_str = re.sub(r'\bya\b', 'yeah', data_str)
	    data_str = re.sub(r'\bcant\b', 'can not', data_str)
	    data_str = re.sub(r'\bdont\b', 'do not', data_str)
	    data_str = re.sub(r'\bwont\b', 'will not', data_str)
	    data_str = re.sub(r'\bid\b', 'i would', data_str)
	    data_str = re.sub(r'wtf', 'what the fuck', data_str)
	    data_str = re.sub(r'\bwth\b', 'what the hell', data_str)
	    data_str = re.sub(r'\br\b', 'are', data_str)
	    data_str = re.sub(r'\bu\b', 'you', data_str)
	    data_str = re.sub(r'\bk\b', 'OK', data_str)
	    data_str = re.sub(r'\bsux\b', 'sucks', data_str)
	    data_str = re.sub(r'\bno+\b', 'no', data_str)
	    data_str = re.sub(r'\bcoo+\b', 'cool', data_str)
	    data_str = re.sub(r'rt\b', '', data_str)
	    data_str = data_str.strip()
	    return data_str
	    
	fix_abbreviation_udf = udf(fix_abbreviation, StringType())     
 
check: 
 .. code-block:: python

	df = df.withColumn('fixed_abbrev',fix_abbreviation_udf(df['text_non_asci']))
	df.show(5,True)

ouput:

.. code-block:: python

	+--------------------+----------+-------+--------------------+--------------------+
	|                text|        id|pubdate|       text_non_asci|        fixed_abbrev|
	+--------------------+----------+-------+--------------------+--------------------+
	|10 Things Missing...|2602860537|  18536|10 Things Missing...|10 things missing...|
	|RT @_NATURALBWINN...|2602850443|  18536|RT @_NATURALBWINN...|@_naturalbwinner ...|
	|RT @HBO24 yo the ...|2602761852|  18535|RT @HBO24 yo the ...|@hbo24 yo the #ne...|
	|Aaaaaaaand I have...|2602738438|  18535|Aaaaaaaand I have...|aaaaaaaand i have...|
	|can I please have...|2602684185|  18535|can I please have...|can i please have...|
	+--------------------+----------+-------+--------------------+--------------------+
	only showing top 5 rows

* remove irrelevant features

.. code-block:: python

	def remove_features(data_str):
	    # compile regex
	    url_re = re.compile('https?://(www.)?\w+\.\w+(/\w+)*/?')
	    punc_re = re.compile('[%s]' % re.escape(string.punctuation))
	    num_re = re.compile('(\\d+)')
	    mention_re = re.compile('@(\w+)')
	    alpha_num_re = re.compile("^[a-z0-9_.]+$")
	    # convert to lowercase
	    data_str = data_str.lower()
	    # remove hyperlinks
	    data_str = url_re.sub(' ', data_str)
	    # remove @mentions
	    data_str = mention_re.sub(' ', data_str)
	    # remove puncuation
	    data_str = punc_re.sub(' ', data_str)
	    # remove numeric 'words'
	    data_str = num_re.sub(' ', data_str)
	    # remove non a-z 0-9 characters and words shorter than 1 characters
	    list_pos = 0
	    cleaned_str = ''
	    for word in data_str.split():
	        if list_pos == 0:
	            if alpha_num_re.match(word) and len(word) > 1:
	                cleaned_str = word
	            else:
	                cleaned_str = ' '
	        else:
	            if alpha_num_re.match(word) and len(word) > 1:
	                cleaned_str = cleaned_str + ' ' + word
	            else:
	                cleaned_str += ' '
	        list_pos += 1
	    # remove unwanted space, *.split() will automatically split on 
	    # whitespace and discard duplicates, the " ".join() joins the 
	    # resulting list into one string.    
	    return " ".join(cleaned_str.split()) 
	# setup pyspark udf function     
	remove_features_udf = udf(remove_features, StringType())  

check: 
 .. code-block:: python

	df = df.withColumn('removed',remove_features_udf(df['fixed_abbrev']))
	df.show(5,True)

ouput:

.. code-block:: python

	+--------------------+----------+-------+--------------------+--------------------+--------------------+
	|                text|        id|pubdate|       text_non_asci|        fixed_abbrev|             removed|
	+--------------------+----------+-------+--------------------+--------------------+--------------------+
	|10 Things Missing...|2602860537|  18536|10 Things Missing...|10 things missing...|things missing in...|
	|RT @_NATURALBWINN...|2602850443|  18536|RT @_NATURALBWINN...|@_naturalbwinner ...|oh and do not lik...|
	|RT @HBO24 yo the ...|2602761852|  18535|RT @HBO24 yo the ...|@hbo24 yo the #ne...|yo the newtwitter...|
	|Aaaaaaaand I have...|2602738438|  18535|Aaaaaaaand I have...|aaaaaaaand i have...|aaaaaaaand have t...|
	|can I please have...|2602684185|  18535|can I please have...|can i please have...|can please have t...|
	+--------------------+----------+-------+--------------------+--------------------+--------------------+
	only showing top 5 rows

4. Sentiment Analysis  main function

.. code-block:: python

	from pyspark.sql.types import FloatType

	from textblob import TextBlob

	def sentiment_analysis(text):
	    return TextBlob(text).sentiment.polarity
	    
	sentiment_analysis_udf = udf(sentiment_analysis , FloatType())    


.. code-block:: python

	df  = df.withColumn("sentiment_score", sentiment_analysis_udf( df['removed'] ))
	df.show(5,True)


* Sentiment score

.. code-block:: python

	+--------------------+---------------+
	|             removed|sentiment_score|
	+--------------------+---------------+
	|things missing in...|    -0.03181818|
	|oh and do not lik...|    -0.03181818|
	|yo the newtwitter...|      0.3181818|
	|aaaaaaaand have t...|     0.11818182|
	|can please have t...|     0.13636364|
	+--------------------+---------------+
	only showing top 5 rows

* Words frequency 

.. figure:: images/sen_word_freq.png
   :align: center


* Sentiment Classification

 .. code-block:: python

	def condition(r):
	    if (r >=0.1):
	        label = "positive" 
	    elif(r <= -0.1):
	        label = "negative"
	    else: 
	        label = "neutral" 
	    return label
	    
	sentiment_udf = udf(lambda x: condition(x), StringType())  

5. Output

*  Sentiment Class

.. figure:: images/sen_class.png
   :align: center

* Top tweets from each sentiment class

 .. code-block:: python

	+--------------------+---------------+---------+
	|                text|sentiment_score|sentiment|
	+--------------------+---------------+---------+
	|and this #newtwit...|            1.0| positive|
	|"RT @SarahsJokes:...|            1.0| positive|
	|#newtwitter using...|            1.0| positive|
	|The #NewTwitter h...|            1.0| positive|
	|You can now undo ...|            1.0| positive|
	+--------------------+---------------+---------+
	only showing top 5 rows

 .. code-block:: python

	+--------------------+---------------+---------+
	|                text|sentiment_score|sentiment|
	+--------------------+---------------+---------+
	|Lists on #NewTwit...|           -0.1|  neutral|
	|Too bad most of m...|           -0.1|  neutral|
	|the #newtwitter i...|           -0.1|  neutral|
	|Looks like our re...|           -0.1|  neutral|
	|i switched to the...|           -0.1|  neutral|
	+--------------------+---------------+---------+
	only showing top 5 rows


 .. code-block:: python 

	+--------------------+---------------+---------+
	|                text|sentiment_score|sentiment|
	+--------------------+---------------+---------+
	|oh. #newtwitter i...|           -1.0| negative|
	|RT @chqwn: #NewTw...|           -1.0| negative|
	|Copy that - its W...|           -1.0| negative|
	|RT @chqwn: #NewTw...|           -1.0| negative|
	|#NewTwitter has t...|           -1.0| negative|
	+--------------------+---------------+---------+
	only showing top 5 rows


N-grams and Correlations
++++++++++++++++++++++++


Topic Model: Latent Dirichlet Allocation
++++++++++++++++++++++++++++++++++++++++



.. _Sentiment analysis: https://en.wikipedia.org/wiki/Sentiment_analysis




