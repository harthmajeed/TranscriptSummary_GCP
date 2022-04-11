from cgitb import text
from youtube_transcript_api import YouTubeTranscriptApi
from flask import Flask, render_template, request

import spacy
#spacy.download('en_core_web_sm')
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def summarize_text(text, per):
    nlp = spacy.load('en_core_web_sm')
    doc= nlp(text)
    tokens=[token.text for token in doc]
    word_frequencies={}
    for word in doc:
        if word.text.lower() not in list(STOP_WORDS):
            if word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1
    max_frequency=max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word]=word_frequencies[word]/max_frequency
    sentence_tokens= [sent for sent in doc.sents]
    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():                            
                    sentence_scores[sent]=word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent]+=word_frequencies[word.text.lower()]
    select_length=int(len(sentence_tokens)*per)
    summary=nlargest(select_length, sentence_scores,key=sentence_scores.get)
    final_summary=[word.text for word in summary]
    summary=''.join(final_summary)
    return summary


def summarize_text_nltk(text):
    stop_words = set (stopwords.words("english"))
    words = word_tokenize(text)
    
    freqTable = dict()
    for word in words:
        word = word.lower()
        if word in stop_words:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1
    
    sentences = sent_tokenize(text)
    sentenceValue = dict()
    for sentence in sentences:
        for word, freq in freqTable.items():
            if word in sentence.lower():
                if sentence in sentenceValue:
                    sentenceValue[sentence] += freq
                else:
                    sentenceValue[sentence] = freq

    sumValues = 0
    for sentence in sentenceValue:
        sumValues += sentenceValue[sentence]
    
    average = int(sumValues / len(sentenceValue))
    print('Average: ', average)

    summary = ''
    for sentence in sentences:
        if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)):
            summary += " " + sentence
    return summary


def summarize_text_nltk_2(text):
    sentence_list = nltk.sent_tokenize(text)

    stopwords = nltk.corpus.stopwords.words('english')

    word_frequencies = {}
    for word in nltk.word_tokenize(text):
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    maximum_frequncy = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)

    sentence_scores = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]

    import heapq
    summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)

    summary = ' '.join(summary_sentences)
    return summary


@app.route('/', methods=['POST'])
def greet():
    vid_id = request.form['video_id']
    data = YouTubeTranscriptApi.get_transcript(vid_id)

    text_transcript = ''
    for dict in data:
        text_transcript += dict['text'] + ' '
    
    text_summarization = summarize_text(text_transcript, 0.05)

    text_summarization_nltk = summarize_text_nltk(text_transcript)

    text_summarization_nltk_2 = summarize_text_nltk_2(text_transcript)

    text2 = """There are many techniques available to genereate extractive summarization to keep it simple, 
        I will be using an unsupervised learning approach to find the sentences similarity and rank them.
        Summarization can be defined as a task of producing a concise and fluent summary while preserving key
        information and overall meaning. One benefit of this will be , you don't need to train and build a model
        prior start using it for your project. It's good to understand Cosine similarity to make the best use of 
        the code you are going to see. Cosine similarity is a measure of similarity between two non-zero vectors
        of an inner product space that measures the cosine of the angle between them. Its measures cosine of the 
        angle between vectors. The angle will be 0 if sentences are similar """

    #print(text2)
    #print(summarize_text_nltk(text2))
    #summarize_text_nltk_2(text2)

    return render_template('output.html', 
        transcript=text_transcript, 
        text_sum=text_summarization, 
        text_sum_nltk=text_summarization_nltk,
        text_sum_nltk_2=text_summarization_nltk_2)


if __name__ == '__main__':
    app.run(host='localhost', port=8080, debug=True)

