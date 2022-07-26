import gensim
import pandas as pd
import re 
import textract

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

vectorizer = TfidfVectorizer(min_df=1)


class ProcessText:

    def __init__(self, loc):
        self.file_loc = loc


    def _parse_pdf(self):
        return textract.process(self.file_loc, method='pdfminer').decode()

    
    def _load_data(self, loc):
        df = pd.read_csv(loc)
        return df


    def _split_paragraph(self, text):
        remove_meta = re.sub(r'[^\w\s]', '', text).strip(' ')
        return re.split('\s*?\n\s*?\n\s*?', remove_meta)


    def _process_text(self, text):
        return [p.replace('\n', ' ').replace('  ', ' ').strip(' ') for p in text if len(p) > 200]


    def _convert_to_df(self, text, file_save):
        df = pd.DataFrame({'paragraph':text})
        df.to_csv(file_save)
        return df

    
    def processed_data(self, save_file):
        text = self.parse_pdf()
        para = self.split_paragraph(text)
        text = self.process_text(para) 
        df = self.convert_to_df(text, save_file)
        return df


class ReturnProcessedText:

    def __init__(self, file_loc):
        self.file_loc = file_loc


    def load_df(self):
        df = pd.read_csv(self.file_loc, index_col=0)
        return df


class ReturnQuestions:

    def __init__(self, file_loc):
        self.file_loc = file_loc 

    
    def load_questions(self):
        with open(self.file_loc, 'r') as f:
            data_list = eval(f.read())
        return data_list

class TFIDFVectorizer:

    def __init__(self, text_data, question):
        self.data = text_data
        self.question = question


    def _get_tfidfs_vocab(self, df, col):
        tf_idf = vectorizer.fit_transform(df[col].values)
        words = vectorizer.get_feature_names_out()
        return tf_idf, words


    def _query_processing(self, vocab, query):
        query_tfidf = vectorizer.fit(vocab)
        query_tfidf = query_tfidf.transform([query])
        return query_tfidf


    def generate_tfidf(self):
        df = ReturnProcessedText(self.data).load_df()
        questions = ReturnQuestions(self.question).load_questions()

        tf_idf, words = self._get_tfidfs_vocab(df, 'paragraph')

        top_tfidf_response = []
        for q in questions:
            query = self._query_processing(words, q[0])
            idx = linear_kernel(query, tf_idf).flatten().argsort()[::-1]
            df_results = df.loc[idx, 'paragraph']
            best_match = df_results.iloc[0]
            top_tfidf_response.append((q[0], best_match))
        return top_tfidf_response


class Doc2vecModel:

    def __init__(self, file_loc, question, model):
        self.file_loc = file_loc
        self.question = question
        self.model = model


    def _process_text(self, file_loc, col, tokens_only=False):
        """
        This function is from the Doc2Vec documentation:
        https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html#sphx-glr-auto-examples-tutorials-run-doc2vec-lee-py
        """
        df = ReturnProcessedText(file_loc).load_df()
        for idx, text in enumerate(df[col]):
            tokens = gensim.utils.simple_preprocess(text)
            if tokens_only:
                yield tokens
            else:
                yield gensim.models.doc2vec.TaggedDocument(text, [idx])


    def _train_doc2vec_model(self, text_data, save_name):
        model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, 
                                               epochs=40)
        model.build_vocab(text_data)
        model.train(text_data, total_examples=model.corpus_count,
                    epochs=model.epochs)
        model.save(save_name)


    def fit_doc2vec(self, file_loc, col, save_file):
        tagged_data = list(self._process_text(file_loc, col))
        model = self._train_doc2vec_model(tagged_data, save_file)
        return model


    def generate_doc2vec(self):
        df = ReturnProcessedText(self.file_loc).load_df()
        model = gensim.models.doc2vec.Doc2Vec.load(self.model)
        questions = ReturnQuestions(self.question).load_questions()

        top_doc2vec = []
        for q in questions:
            query = gensim.utils.simple_preprocess(q[0])
            vec = model.infer_vector(query)
            match = model.dv.most_similar([vec],\
                    topn=len(model.dv))
            top_result = match[0][0]
            df_result = df.loc[top_result, 'paragraph']
            top_doc2vec.append((q[0], df_result))
        return top_doc2vec

        
def main(file_loc, model_loc, question_loc, type='tfidf', train=False):
    if type == 'tfidf':
        return TFIDFVectorizer(file_loc, 
                               question_loc).generate_tfidf()
    else:
        doc2vec = Doc2vecModel(file_loc,
                               question_loc,
                               model_loc)
        if train == True:
           doc2vec.fit_doc2vec(file_loc,
                              'paragraph',
                               model_loc)

        return doc2vec.generate_doc2vec()
        

if __name__ == '__main__':
   vals = main('../data/paragraph.csv',
               '../data/doc2vec_model.model',
               '../data/questions.txt',
                type='tfidf', train=False)