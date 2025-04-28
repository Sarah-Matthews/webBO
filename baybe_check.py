import summit
from summit.benchmarks import get_pretrained_reizman_suzuki_emulator
from summit.benchmarks.experimental_emulator import ReizmanSuzukiEmulator 
from summit.utils.dataset import DataSet
import pandas as pd
import numpy as np
import os
import pathlib

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






emulator = get_pretrained_reizman_suzuki_emulator(case=1)

available_catalysts = {
    "P1-L1",
    "P1-L2",
    "P1-L3",
    "P1-L4",
    "P1-L5",
    "P1-L6",
    "P1-L7",
    "P2-L1",
}

#defining parameter space
parameters = [
    CategoricalParameter(
        name="catalyst",
        values=available_catalysts,
        encoding='OHE'
    ),
    NumericalContinuousParameter(
        name="catalyst_loading",
        bounds=(0.5,2.0),
    ),
    NumericalContinuousParameter(
        name="temperature",
        bounds=(30,110),
    ),
    NumericalContinuousParameter(
        name="t_res",
        bounds=(60,600),
    )
]


#defining search space
searchspace = SearchSpace.from_product(parameters)



target_1 = NumericalTarget(name="yld", mode=f"MAX", bounds=(0,100), transformation="LINEAR")

targets_sobo = [target_1]


objective_sobo = Objective(mode="SINGLE", targets = targets_sobo)
#print('Objective defined (sobo)')




'''
recommender = TwoPhaseMetaRecommender(
    initial_recommender=RandomRecommender(),
    recommender=BotorchRecommender()
)
'''

recommender = TwoPhaseMetaRecommender(
        initial_recommender=RandomRecommender(),
        recommender=SequentialGreedyRecommender(
            surrogate_model=GaussianProcessSurrogate(),
            acquisition_function_cls="qEI",
            allow_repeated_recommendations=False,
            allow_recommending_already_measured=False,
        )
    )


campaign_sobo = Campaign(
    searchspace=searchspace,
    objective=objective_sobo,
    recommender=recommender
    
)

#this function should be general (i.e. mobo & sobo) due to the dynamic target value extraction
def perform_df_experiment(data_df: pd.DataFrame, emulator: ReizmanSuzukiEmulator, objective) -> dict:
    conditions = DataSet.from_df(data_df)
    #print(conditions)
    
    emulator_output = emulator.run_experiments(conditions, return_std=True)
    
    result_df = data_df.copy()

    for target in objective.targets:
        target_name = target.name  # Get the name of the target 
        
        # Find the column corresponding to the target_name in the emulator_output
        if target_name in emulator_output.columns:
            target_value = emulator_output[target_name].values[0]  
            result_df[target_name] = target_value # Add the target to the result DataFrame
        else:
            raise ValueError(f"Target column '{target_name}' not found in emulator output.")


    return result_df

#this function should be general (i.e. mobo & sobo) due to the dynamic target value extraction
def perform_df_experiment_multi(data_df: pd.DataFrame, emulator: ReizmanSuzukiEmulator, objective) -> dict:
    conditions = DataSet.from_df(data_df)
    #print(conditions)
    
    #emulator = get_pretrained_reizman_suzuki_emulator(case=1)
    emulator_output = emulator.run_experiments(conditions, return_std=True)
    
    result_df = data_df.copy()

    for target in objective.targets:
        target_name = target.name  # Get the name of the target 
        
        # Find the column corresponding to the target_name in the emulator_output
        if target_name in emulator_output.columns:
            target_values = emulator_output[target_name].values 
            
            #result_df[target_name] = emulator_output[target_name].values # Add the target to the result DataFrame
            result_df[target_name] = pd.to_numeric(target_values, errors='coerce')
        else:
            raise ValueError(f"Target column '{target_name}' not found in emulator output.")

    #print(result_df)

    return result_df


def run_sobo_loop(
    emulator: summit.benchmarks.experimental_emulator.ReizmanSuzukiEmulator,  
    campaign,  
    iterations: int, 
    initial_conditions_df, 
    ):
        """
        Single-objective bayesian optimisation using the BayBe back end

        emulator: Summit experimental emulator  
        campaign: the campaign defined for the optimisation 
        iterations: the number of cycles/iterations to be completed
        """

        #clear the stored measurements between each trial
        campaign._measurements_exp = pd.DataFrame()

        results_baybe_sobo = []
        cumulative_max_df = pd.DataFrame(columns=["Iteration", "Cumulative Max YLD"])
        times_df_sobo = pd.DataFrame(columns=["Iteration", "Time_taken"])

        print("Starting the SOBO loop...")

        parameter_columns = [param.name for param in searchspace.parameters]
        data_df = pd.DataFrame(columns=parameter_columns)

        #print(f"Initial conditions - randomly generated: {initial_conditions_df}")
   
        target_measurement = perform_df_experiment_multi(initial_conditions_df, emulator, objective=objective_sobo)
        #target_measurement = evaluate_candidates(initial_conditions_df)

        campaign.add_measurements(target_measurement)
 
        
        #record the first step
        results_baybe_sobo.append({
            "iteration": 0,
            "measurements": target_measurement
        })
        #print(results_baybe_sobo)

        #initialising a max.
        cumulative_max_yld = float('-inf')

        for i in range(1, iterations+1):
         
            print(f"Running experiment {i }/{iterations}")
            print('campaign before recs', campaign)
        
            recommended_conditions = campaign.recommend(batch_size=1)
            #print(f"Recommended conditions: {recommended_conditions}")

            data_df = pd.concat([data_df, recommended_conditions], ignore_index=True)

            target_measurement = perform_df_experiment(recommended_conditions, emulator, objective=objective_sobo)
            #target_measurement = evaluate_candidates(candidates=recommended_conditions)
            campaign.add_measurements(target_measurement)
            print('measurements in campaign!',campaign.measurements)
            
            #print(f"Iteration {i} took {(time.time() - t1):.2f} seconds")
        
            #eval_df_sobo = evaluate_candidates(target_measurement)
            new_yld = target_measurement['yld'].values[0]
            print(new_yld)

            if new_yld >  cumulative_max_yld:
                cumulative_max_yld = new_yld
            print(cumulative_max_yld)

            cumulative_max_df = pd.concat([cumulative_max_df, pd.DataFrame([{
            "Iteration": i,
            "Cumulative Max YLD": cumulative_max_yld
            }])], ignore_index=True)

            results_baybe_sobo.append({
                "iteration": i ,
                "measurements": target_measurement
            })

            
       
           
       
            print(campaign)
        
        return campaign.measurements, cumulative_max_df, times_df_sobo # cumulative_max_yld,  #results_baybe_sobo,
    


iterations = 3

initial_conditions = pd.DataFrame({
    'catalyst_loading': [0.905050, 1.047307, 0.570121, 0.996824, 1.899186],
    't_res': [497.510310, 150.671508, 407.747716, 588.822581, 496.130944],
    'temperature': [64.302625, 74.051686, 91.163586, 54.836461, 99.064019],
    'catalyst': ['P1-L1', 'P2-L1', 'P2-L1', 'P1-L4', 'P1-L6']
})

results = run_sobo_loop(emulator, campaign_sobo, iterations, initial_conditions)

print(results)