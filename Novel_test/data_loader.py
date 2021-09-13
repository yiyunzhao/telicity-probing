import os
import pandas as pd
import numpy as np
import random as rd
import pickle


def load_pickle(path):
    with open(path,'rb') as f:
        model = pickle.load(f)
    return model

def transform_answers(lis,resp='for'):
    nums = np.array(lis == resp,dtype=float)
    return nums

def getData(d,norm=True):
    if norm:
        dnorm = normalize_array(d)
    else:
        dnorm = d
    labels = ['second','minute','hour','day','week','month','year','decade']
    instance = int(dnorm.shape[0]/len(labels))
    return dnorm, np.array([[i]*instance for i in labels]).flatten()


class data:
    def __init__(self,data_dir):
        self.data_dir = data_dir
        self.human_response_l1 = pd.read_csv(os.path.join(data_dir,'human/l1.csv')).loc[2:,:]
        self.human_response_l2 = pd.read_csv(os.path.join(data_dir,'human/l2.csv')).loc[2:,:]
        self.num_human = len(self.human_response_l1)+len(self.human_response_l2)
        self.verb = 58
        self.model_types = ['bert_base','bert_large','roberta_base','roberta_large']
        self.model_response = {model:load_pickle(os.path.join(data_dir,'computer/result_'+model+'.pickle')) for model in self.model_types}
        self.time_units = ['second','minute','hour','day','week','month','year','decade']
        self.transitivity = {'human_qid':[0,4,8],'model_qid':[0,2,5],'qlabel':['intrans','trans-animate','trans-inanimate'], 'model_fullid':[0,1,2,3,4,5]}
        self.NP = {'human_qid':[20,16,12], 'model_qid':[6, 7, 8], 'qlabel': ['quantifiable-a','none-quantifiable','quantifable-num'], 'model_fullid':[6,7,8] }
        self.context = {'human_qid':[36, 32], 'model_qid':[20, 21],'qlabel': ['atelic-context','telic-context']}
        self.resultative = {'human_qid':[56, 24, 28, 48, 52], 'model_qid':[9, 10, 11, 12, 14],'qlabel':['base','resultative-structure']}
        self.goal = {'human_qid':[40, 44], 'model_qid':[22, 23], 'qlabel':['goal','base']}
        self.time = ['second','hour','week','year']
        self.categories = {'transitivity': self.transitivity, 'NP':self.NP, 'context': self.context, 'resultative': self.resultative, 'goal':self.goal}


    def _fetch_model_response(self,model, category):
        """Fetch human data from two lists[timeUnit, category, response_number] """
        category_params = self.categories[category]
        FOR = []
        qid = category_params['model_qid']
        qlabel = category_params['qlabel']
        catlen, timelen = len(qlabel), len(self.time_units)
        for q in category_params['model_qid']:
            res, _ = getData(model[q]['result'][:,-1:], False)
            FOR.append(1-res)

        FOR = np.array(FOR)

        if category == 'resultative':
            res_base_com=FOR.reshape(5,8, self.verb)[0,[0,2,4,6],:]
            res_res_com=np.average(FOR.reshape(5,8,self.verb)[1:,[0,2,4,6],:],axis=0)
            response = np.array([res_base_com,res_res_com])
            response = response.transpose(1,0,2)
            response_cat=np.average(response,axis=0)
        else:
            response = np.average(FOR.reshape(catlen,timelen,self.verb,-1),axis=-1)
            response = response[:,[0, 2, 4, 6],:]
            response = response.transpose(1,0,2)
            response_cat =  np.average(response,axis=0)

        itemlen = response_cat.shape[-1]
        categories =  np.array([[i]*itemlen for i in qlabel])

        return response_cat,categories,response


    def _fetch_human_response(self, category):
        """Fetch human data from two lists[timeUnit, category, response_number] """
       
        category_params = self.categories[category]
        qid = category_params['human_qid']
        qlabel = category_params['qlabel']
        response = []
        for i in range(1,5):
            time_qs = []
            for j in qid:
                qn = str(i+j)
                this_time_q= transform_answers(np.concatenate([self.human_response_l1[qn],self.human_response_l2[qn]]))
                time_qs.append(this_time_q)
            response.append(time_qs)
        response = np.array(response)
        if category == 'resultative':
            res_base = response[:,0,:]
            res_res = np.average(response[:,1:,:],axis=1)
            response = np.array([res_base,res_res]).transpose(1,0,2)
            response_cat = np.average(response,axis=0)
        else:
            response_cat = np.average(response, axis=0)
        categories = [[i]*self.num_human for i in qlabel]
        categories = np.array(categories)
        return response_cat, categories, response

    def fetch_category(self, category):
        cat = self.categories[category]
        responses, categories, xlabels = [], [],[]
        
        # human 
        human_response, human_category, _ = self._fetch_human_response(category)
        responses.append(human_response), categories.append(human_category)
        xlabels+= ['human']*self.num_human

        # model
        for model_type, model in self.model_response.items():
            mdresp, mdcat, _ = self._fetch_model_response(model,category)
            responses.append(mdresp)
            categories.append(mdcat)
            xlabels+= [model_type]*mdresp.shape[-1]

        responses = np.hstack(responses)
        categories = np.hstack(categories)
        participant_categories = xlabels*len(cat['qlabel'])

        return responses, categories, participant_categories


    def get_transitivity(self):
        """load transitivity data"""
        return self.fetch_category("transitivity")

         
    def get_NP(self):
        """Load Noun Phrase data """
        return self.fetch_category("NP")

    def get_resultative(self):
        """Load resultative data """
        return self.fetch_category("resultative")

    def get_context(self):
        """Load context data"""
        return self.fetch_category("context")

    def get_goal(self):
        """Load goal data """
        return self.fetch_category("goal")

    def get_time(self):
        """Load temporal data """
        responses, categories, participant_categories =[],[],[]
        human_response = []
        for k in self.categories.keys():
            _, _, resp = self._fetch_human_response(k)
            human_response.append(resp)
        human_response = np.average(np.hstack(human_response),axis=1)
        responses.append(human_response)

        categories.append(np.array([[i]*self.num_human for i in self.time]))
        participant_categories+= ['human']*self.num_human


        for m, model in self.model_response.items():
            model_response = []
            for k in self.categories.keys():
                _, _, resp = self._fetch_model_response(model,k)
                model_response.append(resp)
            model_response = np.average(np.hstack(model_response),axis=1)
            responses.append(model_response)
            categories.append(np.array([[i]*self.verb for i in self.time]))
            
            participant_categories+= [m]*self.verb
        responses = np.hstack(responses)
        categories = np.hstack(categories)
        participant_categories =  participant_categories * len(self.time)
        return responses, categories, participant_categories



