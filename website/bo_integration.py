from baybe import Campaign
from baybe.objective import Objective
from baybe.parameters import CategoricalParameter, NumericalContinuousParameter, NumericalDiscreteParameter, SubstanceParameter
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget, TargetTransformation
from baybe.recommenders import RandomRecommender, SequentialGreedyRecommender, TwoPhaseMetaRecommender
from baybe.surrogates import GaussianProcessSurrogate
from baybe.utils.dataframe import add_fake_results
import pandas as pd
import numpy as np
import torch
import json
import random

from baybe.recommenders.pure.bayesian import base, sequential_greedy
from botorch import fit_gpytorch_mll
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.models.cost import AffineFidelityCostModel
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.acquisition.max_value_entropy_search import qMultiFidelityMaxValueEntropy



def setup_bo(expt_info, target, opt_type, batch_size=1):
    variable_type_dict = pd.read_json(expt_info.variables)
    print('variable_type_dict', variable_type_dict)
    baybe_paramter_list = []
    target = target[0]
    opt_type = opt_type[0]
    print('target:',target, 'opt type', opt_type)

    columns = list(variable_type_dict.keys())
    target_name = columns[int(target)]
    for col in variable_type_dict.columns:
        df = variable_type_dict[col].T
        if df['parameter-type'] == 'subs':
            baybe_paramter_list.append(
                SubstanceParameter(f"{col}", data=df['json'], encoding=f"{df['encoding']}")
            )
        elif df['parameter-type'] == "cat":
            baybe_paramter_list.append(
                CategoricalParameter(f"{col}", values=pd.read_json(df['json'])[f'{col}'], encoding="OHE")
            )
        elif df['parameter-type'] == 'int':
            baybe_paramter_list.append(
                NumericalDiscreteParameter(f"{col}", values=list(np.arange(int(df['min']), int(df['max']), 1.0)))
            )
        else:
            baybe_paramter_list.append(
                NumericalContinuousParameter(f"{col}", bounds=(float(df['min']), float(df['max'])))
            )
    search_space = SearchSpace.from_product(baybe_paramter_list)

    objective = Objective(mode="SINGLE", targets=[NumericalTarget(name=target_name, mode=f"{opt_type}", bounds=[variable_type_dict[target_name]["min"],variable_type_dict[target_name]["max"]], transformation= TargetTransformation.LINEAR)])
  
  
    recommender = TwoPhaseMetaRecommender(
        initial_recommender=RandomRecommender(),
        recommender=SequentialGreedyRecommender(
            surrogate_model=GaussianProcessSurrogate(),
            acquisition_function_cls=f"{expt_info.acqFunc}",
            allow_repeated_recommendations=False,
            allow_recommending_already_measured=False,
        )
    )

    
    

    

    campaign = Campaign(
        searchspace=search_space,
        recommender=recommender,
        objective=objective,
    )
    
    print('campaign empty', campaign)

    print('measurements', expt_info.data)
    print('add measurements')
    data_json = json.loads(expt_info.data)
    data_df = pd.DataFrame(data_json)
    print('data_df', data_df)
    campaign.add_measurements(data_df)

    print('campaign full', campaign)
    rec = campaign.recommend(batch_size=batch_size)
    print('rec in setup_bo', rec)
    print('campaign after .recommend', campaign)
    return campaign, rec #campaign.recommend(batch_size=batch_size)
    


def run_bo(expt_info, batch_size):
    print('expt_info.campaign within run_bo', expt_info.campaign)

    campaign = Campaign.from_config(expt_info.campaign)
 
    print('campaign within run_bo',campaign)
    print('measurements', campaign.measurements)
    rec = campaign.recommend(batch_size=batch_size)
    print('campaign after rec, run_bo', campaign)
    return campaign, rec



def setup_mobo(expt_info, targets,  opt_types, weights, batch_size=1):
    variable_type_dict = pd.read_json(expt_info.variables)
    baybe_parameter_list = []
    columns = list(variable_type_dict.keys())
    targets_arr = []

    for i in range(len(targets)):
        target = targets[i] 
        opt_type = opt_types[i]  
        target_name = columns[int(target)] 
        
        numerical_target = NumericalTarget(
            name=target_name,
            mode=f"{opt_type}",  
            bounds=[variable_type_dict[target_name]["min"], 
                    variable_type_dict[target_name]["max"]],
            transformation=TargetTransformation.LINEAR
        )

        targets_arr.append(numerical_target)

    for col in variable_type_dict.columns:
        df = variable_type_dict[col].T
        if df['parameter-type'] == 'subs':
            baybe_parameter_list.append(
                SubstanceParameter(f"{col}", data=df['json'], encoding=f"{df['encoding']}")
            )
        elif df['parameter-type'] == "cat":
            baybe_parameter_list.append(
                CategoricalParameter(f"{col}", values=pd.read_json(df['json'])[f'{col}'], encoding="OHE")
            )
        elif df['parameter-type'] == 'int':
            baybe_parameter_list.append(
                NumericalDiscreteParameter(f"{col}", values=list(np.arange(int(df['min']), int(df['max']), 1.0)))
            )
        else:
            baybe_parameter_list.append(
                NumericalContinuousParameter(f"{col}", bounds=(float(df['min']), float(df['max'])))
            )

    print(targets_arr)

    search_space = SearchSpace.from_product(baybe_parameter_list)
    
    objective = Objective(mode="DESIRABILITY",
                          targets=targets_arr, 
                          weights=weights, 
                          combine_func=expt_info.combine_func)

    recommender = TwoPhaseMetaRecommender(
        initial_recommender=RandomRecommender(),
        recommender=SequentialGreedyRecommender(
            surrogate_model=GaussianProcessSurrogate(),
            acquisition_function_cls=f"{expt_info.acqFunc}",
            allow_repeated_recommendations=False,
            allow_recommending_already_measured=False,
        )
    )
    campaign = Campaign(
        searchspace=search_space,
        recommender=recommender,
        objective=objective,
    )


    print('campaign empty', campaign)

    print('measurements', expt_info.data)
    print('add measurements')
    data_json = json.loads(expt_info.data)
    data_df = pd.DataFrame(data_json)
    print('data_df', data_df)
    campaign.add_measurements(data_df)

    print('campaign full', campaign)

    return campaign #campaign.recommend(batch_size=batch_size)

    

#MFBO section

def runMes(model, parameter_bounds, fidelities, target_fidelity, target, opt_type, fixed_cost):
    
    #generating candidate set
    candidates = []

    for param in parameter_bounds:
        name, properties = next(iter(param.items()))  
        if target not in name:
            if 'bounds' in properties:  # continuous parameters
                min_val, max_val = properties['bounds']
                values = min_val + (max_val - min_val) * torch.rand(1000)
    
            elif 'values' in properties:  # integer parameter values
                choices = torch.tensor(properties['values'])
                values = choices[torch.randint(0, len(choices), (1000,))]  # Random choice
    
            candidates.append(values)

    candidate_set_no_hf = torch.stack(candidates, dim=1)
    candidate_set = torch.tensor(np.concatenate((candidate_set_no_hf, np.array([[random.choice(fidelities) for x in range(1000)]]).T), axis=1))

    fidelity_index = candidate_set.shape[1] -1
    target_fidelities = {fidelity_index: target_fidelity[0]}  #column containing fidelities and target fidelity number
    
    cost_model = AffineFidelityCostModel(fidelity_weights=target_fidelities, fixed_cost=fixed_cost) 
    
    cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

    if opt_type == 'MAX':
        maximise_bool = True
    else:
        maximise_bool = False

    acquisition = qMultiFidelityMaxValueEntropy(
            model=model,
            cost_aware_utility=cost_aware_utility,
            project=lambda x: project_to_target_fidelity(X=x, target_fidelities=target_fidelities),
            candidate_set=candidate_set,
            maximize=maximise_bool
        )
    

    acquisitionScores = acquisition.forward(candidate_set.reshape(-1, 1, candidate_set.shape[1]))

    return acquisitionScores, candidate_set

def optimiseAcquisitionFunction(model, acq_function,  candidate_set):
    """suggest a new candidate within the search space using MES acquisition function."""

    candidate_idx = torch.argmax(acq_function)
    candidate = candidate_set[candidate_idx].unsqueeze(0)

    posterior = model.posterior(candidate)
    predicted_value = posterior.mean.item()

    fidelity = candidate[0,-1].item()

    return candidate, predicted_value, fidelity

def run_mfbo(expt_info, target, opt_type, batch_size, fidelity_parameters, target_fidelity, fidelity_column, fixed_cost ):
    variable_type_dict = pd.read_json(expt_info.variables)
    parameter_list = []
    target = target[0]
    opt_type = opt_type[0]

    data_list = json.loads(expt_info.data)
    data_df = pd.DataFrame(data_list)
    last_iteration = data_df['iteration'].iloc[-1]

    columns = list(variable_type_dict.keys())
    target_name = columns[int(target)]

    for col in variable_type_dict.columns:
        df = variable_type_dict[col].T
        
        if df['parameter-type'] == 'int':
            parameter_list.append(
            {f"{col}": {'values': list(np.arange(int(df['min']), int(df['max']), 1.0))}})
        elif df['parameter-type'] == 'cont':
            parameter_list.append(
            {f"{col}": {'bounds': (float(df['min']), float(df['max']))}})

    train_obj = torch.tensor(data_df[target_name].astype(float).values).reshape(-1, 1)
    train_x_full = torch.tensor(
    np.hstack((
        data_df.drop(columns=[target_name, 'iteration', fidelity_column]).values.astype(np.float64),
        data_df[fidelity_column].values.reshape(-1, 1)
    )),
    dtype=torch.float64
    )

    model = SingleTaskMultiFidelityGP(train_x_full, train_obj, data_fidelities=[-1])
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    acquisition_function, candidate_set = runMes(model=model, parameter_bounds = parameter_list, fidelities = fidelity_parameters, target_fidelity = target_fidelity, target = target_name, opt_type = opt_type, fixed_cost= fixed_cost)
        
    top_candidate, predicted_value, fidelity = optimiseAcquisitionFunction(model=model, acq_function = acquisition_function, candidate_set = candidate_set)
    iteration = int(last_iteration) +1

    param_columns = [col for col in data_df.columns if col not in {target_name, 'iteration'}]
    candidate_values = top_candidate.numpy().flatten().tolist()

    row = dict(zip(param_columns[:len(candidate_values)], candidate_values))
    row[target_name] = predicted_value 
    row[fidelity_column] = fidelity 
    row['iteration'] = iteration  


    new_recommendation_df = pd.DataFrame([row])
    return new_recommendation_df