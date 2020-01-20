"""
run the web server

$FLASK_ENV=development FLASK_APP=app.py flask run

We use gunicorn to help us handle more incoming request at the same time.

gunicorn -w 3 -b localhost:5000 -t 30 --reload app:app

"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
import logging
import torch
import torch.nn.functional as F
from time import strftime, localtime
from config import Configs
from pytorch_pretrained_bert import BertModel
from data_utils import Tokenizer4Bert
from models.bert_ssc import BERT_SSC
from utils import remove_delimiter,remove_separator,remove_empty,remove_two_spaces,remove_three_spaces
from flask import Flask, jsonify, request
from pydantic import BaseModel, Schema, Field
from flaskerk import Flaskerk
from enum import Enum

class Data(BaseModel):
    """
    This is the description of the data model.
    """
    text: str = Schema(..., max_length=140, description='this is a input sentence')  # ... means no default value

    class Config:
        schema_extra = {
            'examples': [
                {
                    'text': 'a input sentence',
                }
            ]
        }

class Output(str, Enum):
    not_relevant = "Not Relevant"
    relevant = "Relevant"
    other = "Can't Decide"


class Response(BaseModel):
    """
    basic response model
    """
    pred: str
    #pred: Output = Field(..., alias='Output', description = "output possiblities")

app = Flask(__name__)
# api documentation & validation of request and response
api = Flaskerk(app)  
# Initialize logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

class TextInModel:
    """Load the pre-trained model, you can use your model just as easily.
    """
    def __init__(self, opt):
        self.opt = opt
        if 'bert' in opt.model_name:
            self.tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
            bert = BertModel.from_pretrained(opt.pretrained_bert_name)
            self.model = BERT_SSC(bert, opt).to(opt.device)
            logger.info('loading model {0} ... done'.format(opt.model_name))
            # remember removed map_location='cpu' when using on server w GPU
            self.model.load_state_dict(torch.load(opt.state_dict_path))

            # switch model to evaluation mode
            self.model.eval()
            torch.autograd.set_grad_enabled(False)
        else:
            logger.info('Now, we only support bert-based model')
            raise ValueError("Now, we only support bert-based model")

    def predict_prob(self, batch_raw_texts):
        """
        batch preprocessing can efficiently bosst qps due to using gpu's nature.
        

        paras:
            raw_texts: list of string
        """
        # text-preprocessing
        batch_raw_texts = [remove_delimiter(raw_text) for raw_text in batch_raw_texts]
        batch_raw_texts = [remove_separator(raw_text) for raw_text in batch_raw_texts]
        batch_raw_texts = [remove_empty(raw_text) for raw_text in batch_raw_texts]
        batch_raw_texts = [remove_two_spaces(raw_text) for raw_text in batch_raw_texts]
        batch_raw_texts = [remove_three_spaces(raw_text) for raw_text in batch_raw_texts]
        # tokenize
        text_bert_indices = []
        for text in batch_raw_texts:
            ls_tokens = self.tokenizer.text_to_sequence("[CLS] " + text)
            text_bert_indices.append(ls_tokens)
        # conver to tensor
        text_bert_indices = torch.tensor(text_bert_indices, dtype=torch.int64).to(self.opt.device)

        t_inputs = [text_bert_indices]
        t_outputs = self.model(t_inputs)

        t_probs = F.softmax(t_outputs, dim=-1).cpu().detach().numpy()
        return t_probs



@app.route('/predict', methods=['POST'])
@api.validate(data=Data, resp=Response)
def predict():
    if request.method == 'POST':
        text = request.get_json()["text"]
        # pred
        t_probs = model.predict_prob(
            [text]  
            )
        pred = t_probs.argmax(axis=-1)
        return Response(pred=Configs.polarity_dict[pred[0]])

if __name__ == '__main__':
    # Model Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert_ssc', type=str, required = True)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--max_seq_len', default=140, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--state_dict_path', default="../state_dict/bert_ssc_val_acc0.5", type=str, help='path to persist model')
    opt = parser.parse_args()
    # loading model
    log_file = '../logs/serve_{}-{}.log'.format(opt.model_name, strftime("%y%m%d-%H%M", localtime()))
    if not os.path.exists('../logs'):
        os.mkdir('../logs')
    logger.addHandler(logging.FileHandler(log_file))

    logger.info("Loading PyTorch model and Flask starting server ...")
    logger.info("Please wait until server has fully started")

    model = TextInModel(opt)
    app.run(host = '0.0.0.0')

