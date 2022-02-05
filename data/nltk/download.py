import nltk


def download_nltk_data(resource_name, port=None):
    if port is not None:
        nltk.set_proxy("https://127.0.0.1:%s/" % port)
    nltk.download(resource_name, download_dir='./')


if __name__ == '__main__':
    download_nltk_data("stopwords")
    download_nltk_data("wordnet")
    download_nltk_data("punkt")
