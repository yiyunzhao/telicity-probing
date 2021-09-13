import pandas as pd
import numpy as np
import os


def mean_of_category(model,questions):
    human_for = model.loc[:,questions] == 'for'
    human_for_mean = human_for.mean(axis=1)
    return human_for_mean

def computer_mean_of_category(model,questions):
    model_for = model.loc[questions,['binary_prediction']]=='for'
    meanfor=model_for.mean(axis=0)[0]
    return meanfor

class data:
    def __init__(self,data_dir):
        self.data_dir = data_dir
        self.verb = {'atelic': range(6,20), 'telic': range(20,34)}
        self.article = {'the':[34,39,44,49,57,63],'a':[35,42,47,52,55,61],'mass':[37,41,46,50,54,59],'num':[36,40,45,51,58,60],'other-quant':[38,43,48,53,56,62]}
        self.resultative = {'base':[64,67,69,71,72,74,76,78,80],'res':[65,66,68,70,73,75,77,79,81]}
        self.context  = {'contextin':[83,85,86,89,92,94,97,99],'contextfor':[82,84,87,88,93,95,96,98]}
        self.human_response = pd.read_csv(os.path.join(data_dir,'Telicity_Probing_Final.csv')).loc[2:,[str(i) for i in range(1,100)]]
        self.model_types = ['bert-base','bert-large','roberta-base','roberta-large']
        self.num_human = 59
        self.model_response = {model:pd.read_csv(os.path.join(data_dir,model+'-results-qualtrics.csv')) for model in self.model_types}

    def get_verb(self):
        """load verb data"""

        #human
        human_verb_for=self.human_response.loc[:,[str(i) for i in self.verb['atelic']]] == 'for'
        human_verb_in=self.human_response.loc[:,[str(i) for i in self.verb['telic']]] == 'for'
        human_verb_in_mean = human_verb_in.mean(axis=1)
        human_verb_for_mean = human_verb_for.mean(axis=1)
        human_verb_resp = pd.concat([human_verb_for_mean,human_verb_in_mean])
        verb_type =['atelic-type']*self.num_human + ['telic-type']*self.num_human 

        #model
        model_line = {}
        for model, modelcsv in self.model_response.items():
            model_line[model]=[computer_mean_of_category(modelcsv,[i-1 for i in v]) for v in self.verb.values()]
        return verb_type, human_verb_resp, model_line
        
    def get_NP(self):
        """Load Noun Phrase data """
        
        #human
        thefor= mean_of_category(self.human_response,[str(i) for i in self.article['the']])
        afor=mean_of_category(self.human_response,[str(i) for i in self.article['a']])
        massfor=mean_of_category(self.human_response,[str(i) for i in self.article['mass']])
        numfor=mean_of_category(self.human_response,[str(i) for i in self.article['num']])
        plurfor=mean_of_category(self.human_response,[str(i) for i in self.article['other-quant']])

        human_article_resp = pd.concat([thefor,afor,massfor,numfor,plurfor])
        article_type = ['quant-the']*self.num_human+['quant-a']*self.num_human+ ['non-quant']*self.num_human+ ['quant-num']*self.num_human+['other-quant']*self.num_human

        model_line = {}
        for model, modelcsv in self.model_response.items():
            model_line[model]=[computer_mean_of_category(modelcsv,[i-1 for i in v]) for v in self.article.values()]
        return article_type, human_article_resp, model_line

    def get_resultative(self):
        """Load resultative data """

        resbasefor= mean_of_category(self.human_response,[str(i) for i in self.resultative['base']])
        resresfor=mean_of_category(self.human_response,[str(i) for i in self.resultative['res']])
        human_res_resp = pd.concat([resbasefor,resresfor])
        res_type = ['base']*self.num_human + ['resultative']*self.num_human 
        
        model_line = {}
        for model, modelcsv in self.model_response.items():
            model_line[model]=[computer_mean_of_category(modelcsv,[i-1 for i in v]) for v in self.resultative.values()]
        return res_type, human_res_resp, model_line


    def get_context(self):
        """Load context data"""

        contextin= mean_of_category(self.human_response,[str(i) for i in self.context['contextin']])
        contextfor=mean_of_category(self.human_response,[str(i) for i in self.context ['contextfor']])
        human_context_resp = pd.concat([contextin,contextfor])
        context_type =  ['context.in']*self.num_human +['context.for']*self.num_human
        model_line = {}
        for model, modelcsv in self.model_response.items():
            model_line[model]=[computer_mean_of_category(modelcsv,[i-1 for i in v]) for v in self.context.values()]

        return context_type, human_context_resp, model_line
