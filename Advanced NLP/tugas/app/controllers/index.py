import string
import stanza
from flask import Blueprint, render_template, request

index_bp = Blueprint('index', __name__)

STOP_WORDS = {
    "dan", "di", "ke", "yang", "untuk", "dari", "pada", "dengan", "sebagai",
    "itu", "ini", "ada", "karena", "juga", "atau", "saat", "akan", "tidak"
}

nlp = None


def get_nlp():
    global nlp
    if nlp is None:
        stanza.download('id')
        nlp = stanza.Pipeline('id', processors='tokenize,pos,lemma')
    return nlp


def preprocess_text(text):
    if not text.strip():
        return {
            "original_text": "",
            "tokens": [],
            "filtered_tokens": [],
            "lemmatized_tokens": [],
            "pos_tagged_tokens": []
        }

    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))

    doc = get_nlp()(text)

    tokens, lemmas, pos_tags = [], [], []
    for sentence in doc.sentences:
        for word in sentence.words:
            tokens.append(word.text)
            lemmas.append(word.lemma)
            pos_tags.append((word.text, word.upos))

    filtered_tokens = [
        w for w in tokens if w not in STOP_WORDS and w.isalnum()]

    return {
        "original_text": text,
        "tokens": tokens,
        "filtered_tokens": filtered_tokens,
        "lemmatized_tokens": lemmas,
        "pos_tagged_tokens": pos_tags
    }


@index_bp.route('/', methods=['GET', 'POST'])
def index():
    requested_data = ''
    result = None

    if request.method == 'POST':
        requested_data = request.form.get('inputText', '')
        result = preprocess_text(requested_data)

    return render_template('index/index.html', data=result)
