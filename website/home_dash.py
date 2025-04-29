from flask import Blueprint, render_template, request, redirect, url_for, jsonify, session, flash, make_response
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from . import db 
import re
import random
import json
from .models import Data, Experiment, Target, Fidelity
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly
from flask import session
from summit.benchmarks import get_pretrained_reizman_suzuki_emulator
from summit.utils.dataset import DataSet
from .bo_integration import setup_bo,setup_mobo, run_bo, run_mfbo

from baybe import Campaign

home_dash = Blueprint("home_dash", __name__)


@home_dash.route('/tutorial', methods=['GET'])
@login_required
def tutorial():
    return render_template('tutorial.html', user=current_user)

@home_dash.route('/video_tutorial', methods=['GET'])
@login_required
def video_tutorial():
    return render_template('video_tutorial.html', user=current_user)

@home_dash.route('/explanations', methods=['GET'])
@login_required
def explanations():
    return render_template('explanations.html', user=current_user)


@home_dash.route("/", methods=["GET", "POST"])
@login_required
def home():
    if request.method == "POST":
        if request.form['action'] == "add-dataset":
            if Data.query.count() == 3:
                flash("Whoops! You cannot have more than 3 datasets uploaded. Please export and delete at least one dataset in your repository.", category="error")
            else:
                experiment_type = request.form.get("experiment_type")
                experiment_fidelities = request.form.get("experiment_fidelities")
                if experiment_type == "single-objective" and experiment_fidelities == "single-fidelity":
                    return redirect(url_for("experiment_forms.setup"))
                elif experiment_type == "single-objective" and experiment_fidelities == "multi-fidelity":
                    return redirect(url_for("experiment_forms.setup_mfbo"))
                elif experiment_type == "multi-objective" and experiment_fidelities == "single-fidelity":
                    return redirect(url_for("experiment_forms.setup_mo"))
                return redirect(url_for("dataset_forms.select_upload_method"))
        elif request.form['action'] == "add-experiment":
            if not db.session.query(Data).all():
                flash("Whoops! You need to upload a dataset first!", category="error")
            else:
                experiment_type = request.form.get("experiment_type")
                experiment_fidelities = request.form.get("experiment_fidelities")
                if experiment_type == "single-objective" and experiment_fidelities == "single-fidelity":
                    return redirect(url_for("experiment_forms.setup"))
                elif experiment_type == "single-objective" and experiment_fidelities == "multi-fidelity":
                    return redirect(url_for("experiment_forms.setup_mfbo"))
                elif experiment_type == "multi-objective" and experiment_fidelities == "single-fidelity":
                    return redirect(url_for("experiment_forms.setup_mo"))
                
        elif request.form['action'] == "submit-experiment":
            experiment_type = request.form.get("experiment_type")
            experiment_fidelities = request.form.get("experiment_fidelities")
            if experiment_type == "single-objective" and experiment_fidelities == "single-fidelity":
                return redirect(url_for("experiment_forms.setup"))
            elif experiment_type == "single-objective" and experiment_fidelities == "multi-fidelity":
                return redirect(url_for("experiment_forms.setup_mfbo"))
            elif experiment_type == "multi-objective" and experiment_fidelities == "single-fidelity":
                return redirect(url_for("experiment_forms.setup_mo"))
            
        elif "viewdata-" in request.form['action']:
            session['viewdata'] = request.form['action'].removeprefix('viewdata-')
            return redirect(url_for("home_dash.view_dataset"))
        elif "viewexpt-" in request.form['action']:
            session['viewexpt'] = request.form['action'].removeprefix('viewexpt-')
            return redirect(url_for("home_dash.view_experiment", expt_name=request.form['action'].removeprefix('viewexpt-')))
        elif request.form['action'] == "add-sample-dataset":
            sample_dataset_name = request.form.get('sample-dataset-name')
            if sample_dataset_name == "sample-reizman-suzuki":
                flash("Please select another name and try again.", category="error")
            else:
                please_add_sample_dataset(sample_dataset_name)
            return redirect(url_for("home_dash.home"))
        
        elif request.form['action'] == "add-sample-dataset-mo":
            sample_dataset_name = request.form.get('sample-dataset-name')
            if sample_dataset_name == "sample-reizman-suzuki-mo":  
                flash("Please select another name and try again.", category="error")
            else:
                please_add_sample_dataset_mo(sample_dataset_name) 
            return redirect(url_for("home_dash.home"))
        
        elif request.form['action'] == "add-sample-dataset-mfbo":
            sample_dataset_name = request.form.get('sample-dataset-name-mfbo')
            if sample_dataset_name == "sample-reizman-suzuki-mfbo":  
                flash("Please select another name and try again.", category="error")
            else:
                please_add_sample_dataset_mfbo(sample_dataset_name) 
            return redirect(url_for("home_dash.home"))

        elif "remove-dataset-" in request.form['action']:
            note = Data.query.get(int(request.form['action'].removeprefix("remove-dataset-")))
            db.session.delete(note)
            db.session.commit()
        elif "remove-experiment-" in request.form['action']:
            note = Experiment.query.get(int(request.form['action'].removeprefix("remove-experiment-")))
            db.session.delete(note)
            db.session.commit()
        elif request.form['action'] == "logout":
            return redirect(url_for("auth.logout"))
        elif request.form['action'] == "reset-user":
            Data.query.filter_by(user_id=current_user.id).delete()
            Experiment.query.filter_by(user_id=current_user.id).delete()

    return render_template("home.html", user=current_user)



@home_dash.route("/view_experiment/<string:expt_name>", methods=["POST", "GET"])
@login_required
def view_experiment(expt_name):
   
    
    expt_info = [row for row in Experiment.query.filter_by(name=expt_name).all()][0]
    data_info = Data.query.filter_by(name=expt_info.dataset_name).first()
    targets = Target.query.filter_by(experiment_id=expt_info.id).all()
    df = pd.read_json(expt_info.data)
    variables_df = pd.read_json(expt_info.variables)
    variable_list = list(variables_df.columns)
    
    target_column_names=[]
    target_indices=[]
    target_opt_types = []
    target_weights = []
    
    

    
    for target in targets:
        target_column_names.append(variable_list[target.index])  
        target_indices.append(int(target.index))
        target_opt_types.append(target.opt_type)
        target_weights.append(float(target.weight))
    
    for col in target_column_names:
        df[col] = df[col].apply(lambda x: f'{x}')

    fidelity_params = []
    target_fidelity = []
   
    
    if expt_info.fidelity == 'MULTI':
        fidelities = Fidelity.query.filter_by(experiment_id=expt_info.id).all()
        fidelity_column = data_info.fidelity_column
        
        for fidelity in fidelities:
            fidelity_params.append(fidelity.fidelity_parameter)
            fixed_cost = fidelity.fixed_cost
            if fidelity.target_fidelity == 'True':
                target_fidelity.append(fidelity.fidelity_parameter)


    recs = pd.read_json(expt_info.next_recs)

    if request.method == "POST":
        if request.form['action'] == "view-my-stuff":
            return redirect(url_for('home_dash.home'))
        if request.form['action'] == 'run':
            if expt_info.fidelity == 'SINGLE':

                if expt_info.objective == "SINGLE":
                # sobo
                    if expt_info.iterations_completed == 0:
                        campaign = setup_bo(expt_info,  target=target_indices, opt_type=target_opt_types,  batch_size=expt_info.batch_size)
                        expt_info.campaign = campaign.to_json()
                        campaign, recs = run_bo(expt_info, batch_size=expt_info.batch_size)
                        recs['iteration'] = df['iteration'].max() + 1
                        expt_info.next_recs = recs.to_json()
                    else:
                        campaign, recs = run_bo(expt_info, batch_size=expt_info.batch_size)
                        expt_info.campaign = campaign.to_json()
                        recs['iteration'] = df['iteration'].max() + 1
                        expt_info.next_recs = recs.to_json()
                        


                else:
                # mobo
                    if expt_info.iterations_completed == 0:
                        campaign = setup_mobo(expt_info,  targets=target_indices, opt_types=target_opt_types, weights=target_weights,  batch_size=expt_info.batch_size)
                        expt_info.campaign = campaign.to_json()
                        campaign, recs = run_bo(expt_info, batch_size=expt_info.batch_size)
                        recs['iteration'] = df['iteration'].max() + 1
                        expt_info.next_recs = recs.to_json()
                        
                    else:
                        campaign, recs = run_bo(expt_info, batch_size=expt_info.batch_size)
                        expt_info.campaign = campaign.to_json()
                        recs['iteration'] = df['iteration'].max() + 1
                        expt_info.next_recs = recs.to_json()
                        

                

                

                

            elif expt_info.fidelity == 'MULTI':

                recs = run_mfbo(expt_info, 
                                          
                        target=target_indices, 
                    
                        opt_type=target_opt_types, 

                        batch_size=expt_info.batch_size,

                        fidelity_parameters = fidelity_params,

                        target_fidelity = target_fidelity,
                        
                        fidelity_column = fidelity_column,

                        fixed_cost=fixed_cost)
                
            
                expt_info.next_recs = recs.to_json()
            
            expt_info.iterations_completed = expt_info.iterations_completed + 1
            expt_info.data = df.to_json(orient='records')
            db.session.add(expt_info)
            db.session.flush()
            db.session.commit()
            return redirect(url_for('experiment_forms.run_expt', expt_name=expt_info.name))
        elif request.form['action'] == 'add':
            return redirect(url_for('experiment_forms.add_measurements', expt_name=expt_info.name))
        elif request.form['action'] == 'send':
            return redirect(url_for('dataset_forms.send', expt_name=expt_info.name))
        elif request.form['action'] == 'download':
            csv = df.to_csv(index=False)
            

        
            response = make_response(csv)
            response.headers['Content-Disposition'] = f'attachment; filename={expt_name}.csv'
            response.headers['Content-Type'] = 'text/csv'

            return response

    return render_template(
        'view_experiment.html',
        user=current_user,
        expt_name=expt_info.name, 
        dataset_name=expt_info.dataset_name,
        target_names=target_column_names,
        df=df, 
        max_iteration =  df['iteration'].max() ,
        titles=df.columns.values,
        fidelity = expt_info.fidelity
    )




@home_dash.route("/get_plot_data", methods=["GET"])
@login_required
def get_plot_data():
    x_var = request.args.get("x_var")
    y_var = request.args.get("y_var")

    if not x_var or not y_var:
        return jsonify({"error": "Invalid selection"}), 400

    expt_name = request.args.get("expt_name") 
    expt = Experiment.query.filter_by(name=expt_name).first()
    data_info = Data.query.filter_by(name=expt.dataset_name).first()

    if not expt:
        return jsonify({"error": "Experiment not found"}), 404

    df = pd.read_json(expt.data)

    def is_numeric(val):
        return bool(re.fullmatch(r"^-?\d*\.?\d+$", str(val)))

    df_float = df.copy()
    for col in df.columns:
        df_float[col] = [float(val) if is_numeric(val) else val for val in df[col]]

    if x_var not in df_float.columns or y_var not in df_float.columns:
        return jsonify({"error": "Selected variables not found"}), 400

  
    fig = go.Figure()
    if expt.fidelity == "MULTI": #colouring points by fidelity parameter 
        fidelity_col = data_info.fidelity_column 
        unique_fidelity_values = df_float[fidelity_col].unique()
        colors = px.colors.qualitative.Set1  
        
        color_map = {val: colors[i % len(colors)] for i, val in enumerate(unique_fidelity_values)}
        
        for val in unique_fidelity_values:
            subset = df_float[df_float[fidelity_col] == val]
            fig.add_trace(go.Scatter(
                x=subset[x_var],
                y=subset[y_var],
                mode='markers',
                marker=dict(color=color_map[val]),
                name=f"Fidelity {val}"
            ))
    else:
        fig.add_trace(go.Scatter(
            x=df_float[x_var],
            y=df_float[y_var],
            mode='markers',
            name=f'{x_var.title()} vs {y_var.title()}'
        ))
    
    fig.update_layout(
        xaxis_title=f"{x_var.title()}",
        yaxis_title=f"{y_var.title()}",
        title=f"{x_var.title()} vs {y_var.title()}",
        font=dict(family="Courier New, monospace", size=18, color="RebeccaPurple"),
        height=600,
        width=900,
        autosize=True
    )

    return jsonify({"graph": json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)})




@home_dash.route("/view_dataset", methods=["POST", "GET"])
@login_required
def view_dataset():
    df = [pd.read_json(row.data) for row in Data.query.filter_by(name=session['viewdata']).all()][0]

    if request.method == "POST":
        if request.form['action'] == "view-my-stuff":
            return redirect(url_for('home_dash.home'))
        if request.form['action'] == 'download':
            csv = df.to_csv(index=False)
            # Create response
            response = make_response(csv)
            
            response.headers['Content-Disposition'] = f"attachment; filename={session['viewdata']}.csv"

            response.headers['Content-Type'] = 'text/csv'

            return response
        if request.form['action'] == 'setup-experiment':
            existing_experiment = Experiment.query.filter_by(name=session['viewdata']).first()
            
            
            if existing_experiment:
                print(f"Experiment with name {session['viewdata']} already exists.")
                return redirect(url_for('home_dash.view_experiment', expt_name=f"{session['viewdata']}"))


            if session['viewdata'].endswith("-sample-reizman"):
                variable_types={
                    "catalyst": {"parameter-type":"cat",
                            "json": '[{"catalyst":"P1-L1"},{"catalyst":"P2-L1"},{"catalyst":"P1-L2"},{"catalyst":"P1-L3"}, {"catalyst":"P1-L4"},{"catalyst":"P1-L5"},{"catalyst":"P1-L6"},{"catalyst":"P1-L7"}]',
                    },
                    "t_res": {"parameter-type": "cont", "min": 60.0, "max": 600.0},
                    "temperature": {"parameter-type": "cont", "min": 30.0, "max": 110.0},
                    "catalyst_loading": {"parameter-type": "cont", "min": 0.5, "max": 2.5},
                    "yield": {"parameter-type": "cont", "min": 0.0, "max": 100.0},
                }
                sample_experiment = Experiment(
                    name=f"{session['viewdata']}",
                    dataset_name=f"{session['viewdata']}",
                    data=df.to_json(orient="records"),
                    objective = 'SINGLE',
                    fidelity = 'SINGLE',
                    n_targets = 1,
                    variables=json.dumps(variable_types),
                    kernel="Matern",
                    acqFunc="PI",
                    batch_size=1,
                    next_recs=pd.DataFrame().to_json(orient="records"),
                    iterations_completed=0,
                    user_id=current_user.id,
                )
                db.session.add(sample_experiment)
                db.session.flush()
                db.session.commit()

                targets = Target(
                    index = 4,
                    name = 'yield',
                    opt_type = "MAX", 
                    weight = float(1.0),
                    experiment_id = sample_experiment.id
                )
                db.session.add(targets)
                db.session.commit()


                return redirect(url_for('home_dash.view_experiment', expt_name=f"{session['viewdata']}"))
            elif session['viewdata'].endswith("-sample-reizman-mo"):
                variable_types={
                    "catalyst": {"parameter-type":"cat",
                            "json": '[{"catalyst":"P1-L1"},{"catalyst":"P2-L1"},{"catalyst":"P1-L2"},{"catalyst":"P1-L3"}, {"catalyst":"P1-L4"},{"catalyst":"P1-L5"},{"catalyst":"P1-L6"},{"catalyst":"P1-L7"}]',
                    },
                    "t_res": {"parameter-type": "cont", "min": 60.0, "max": 600.0},
                    "temperature": {"parameter-type": "cont", "min": 30.0, "max": 110.0},
                    "catalyst_loading": {"parameter-type": "cont", "min": 0.5, "max": 2.5},
                    "yield": {"parameter-type": "cont", "min": 0.0, "max": 100.0},
                    "ton": {"parameter-type": "cont", "min": 0.0, "max": 100.0}
                }
                sample_experiment = Experiment(
                    name=f"{session['viewdata']}",
                    dataset_name=f"{session['viewdata']}",
                    data=df.to_json(orient="records"),
                    objective = 'MULTI',
                    fidelity = 'SINGLE',
                    n_targets = 2,
                    variables=json.dumps(variable_types),
                    combine_func = 'MEAN',
                    kernel="Matern",
                    acqFunc="PI",
                    batch_size=1,
                    next_recs=pd.DataFrame().to_json(orient="records"),
                    iterations_completed=0,
                    user_id=current_user.id,
                )
                db.session.add(sample_experiment)
                db.session.flush()
                db.session.commit()

                targets = [
                    Target(index = 4, 
                           name = 'yield', 
                           opt_type = "MAX", 
                           weight = float(1.0), 
                           experiment_id = sample_experiment.id),
                           
                    Target(index = 5, 
                           name = 'ton', 
                           opt_type = "MAX", 
                           weight = float(1.0), 
                           experiment_id = sample_experiment.id)
                ]
                db.session.add_all(targets)
                db.session.commit()

                
                return redirect(url_for('home_dash.view_experiment', expt_name=f"{session['viewdata']}"))
            elif session['viewdata'].endswith("-sample-reizman-mfbo"):
                variable_types={
                   # "catalyst": {"parameter-type":"cat",
                    #        "json": '[{"catalyst":"P1-L1"}]',
                   # },
                    "t_res": {"parameter-type": "cont", "min": 60.0, "max": 600.0},
                    "temperature": {"parameter-type": "cont", "min": 30.0, "max": 110.0},
                    "catalyst_loading": {"parameter-type": "cont", "min": 0.5, "max": 2.5},
                    "yield": {"parameter-type": "cont", "min": 0.0, "max": 100.0},
                }
                sample_experiment = Experiment(
                    name=f"{session['viewdata']}",
                    dataset_name=f"{session['viewdata']}",
                    data=df.to_json(orient="records"),
                    objective = 'SINGLE',
                    fidelity = 'MULTI',
                    n_targets = 1,
                    variables=json.dumps(variable_types),
                    kernel="mfgp",
                    acqFunc="qMultiFidelityMaxValueEntropy",
                    batch_size=1,
                    next_recs=pd.DataFrame().to_json(orient="records"),
                    iterations_completed=0,
                    user_id=current_user.id,
                )
                db.session.add(sample_experiment)
                db.session.flush()
                db.session.commit()

                targets = Target(
                    index = 3,
                    name = 'yield',
                    opt_type = "MAX", 
                    weight = float(1.0),
                    experiment_id = sample_experiment.id
                )
                db.session.add(targets)
                db.session.commit()

                fidelities = [
                    Fidelity(
                    fidelity_parameter = float(1.0),
                    target_fidelity = 'True',
                    fixed_cost = 0.0,
                    experiment_id=sample_experiment.id
                ),
                           
                    Fidelity(
                    fidelity_parameter = float(0.01),
                    target_fidelity = 'False',
                    fixed_cost = 0.0,
                    experiment_id=sample_experiment.id
                )
                ]
                db.session.add_all(fidelities)
                db.session.commit()
                
                
                return redirect(url_for('home_dash.view_experiment', expt_name=f"{session['viewdata']}"))
    return render_template(
        'view_dataset.html',
        user=current_user,
        name=session['viewdata'],
        tables=[df.to_html(classes='data', index=False)],
        titles=df.columns.values,
        summaries = [df.drop(columns=['iteration']).describe().to_html(classes='data', index=True)], #removing iteration column from summary statistics
        summary_titles=df.describe().columns.values,

    )





@home_dash.route('/add-sample-dataset', methods=['POST'])
def add_sample_dataset():
    sample_dataset = {
        "catalyst": ["P1-L3"], "t_res": [600], "temperature": [30],"catalyst_loading": [0.498],
    }
    
    dataset_df = pd.DataFrame(sample_dataset)

    emulator = get_pretrained_reizman_suzuki_emulator(case=1)
    conditions = DataSet.from_df(dataset_df)
    emulator_output = emulator.run_experiments(conditions, rtn_std=True)
    rxn_yield = emulator_output.to_numpy()[0, 5]
    
    dataset_df['yield'] = rxn_yield#*100
    dataset_df['iteration'] = 0

    variable_df = pd.DataFrame(dataset_df.columns, columns=["variables"])
    sample_data = Data(
        name="sample-reizman-suzuki",
        data=dataset_df.to_json(orient="records"),
        variables=variable_df.to_json(orient="records"),
        fidelity = 'SINGLE',
        user_id=current_user.id,
    )
    db.session.add(sample_data)
    db.session.flush()
    db.session.commit()
    return jsonify({})


def please_add_sample_dataset(name):
    sample_dataset = {
        "catalyst": ["P1-L3"], "t_res": [600], "temperature": [30],"catalyst_loading": [0.498],
    }

    dataset_df = pd.DataFrame(sample_dataset)

    emulator = get_pretrained_reizman_suzuki_emulator(case=1)
    conditions = DataSet.from_df(dataset_df)
    emulator_output = emulator.run_experiments(conditions, rtn_std=True)
    rxn_yield = emulator_output.to_numpy()[0, 5]

    dataset_df['yield'] = rxn_yield*100
    dataset_df['iteration'] = 0

    variable_df = pd.DataFrame(dataset_df.columns, columns=["variables"])
    sample_data = Data(
        name=f"{name}-sample-reizman", #"sample-reizman-suzuki",
        data=dataset_df.to_json(orient="records"),
        variables=variable_df.to_json(orient="records"),
        fidelity = 'SINGLE',
        user_id=current_user.id,
    )
    db.session.add(sample_data)
    db.session.flush()
    db.session.commit()



def add_sample_dataset_mo():
    sample_dataset = {
        "catalyst": ["P1-L3"], "t_res": [600], "temperature": [30],"catalyst_loading": [0.498],
    }
    
    dataset_df = pd.DataFrame(sample_dataset)

    emulator = get_pretrained_reizman_suzuki_emulator(case=1)
    conditions = DataSet.from_df(dataset_df)
    emulator_output = emulator.run_experiments(conditions, rtn_std=True)
    rxn_yield = emulator_output.to_numpy()[0, 5]
    rxn_ton = emulator_output.to_numpy()[0, 6]
    
    dataset_df['yield'] = rxn_yield#*100
    dataset_df['TON'] = rxn_ton#*100
    dataset_df['iteration'] = 0

    variable_df = pd.DataFrame(dataset_df.columns, columns=["variables"])
    sample_data = Data(
        name="sample-reizman-suzuki-mo",
        data=dataset_df.to_json(orient="records"),
        variables=variable_df.to_json(orient="records"),
        fidelity = 'SINGLE',
        user_id=current_user.id,
    )
    db.session.add(sample_data)
    db.session.flush()
    db.session.commit()
    return jsonify({})


def please_add_sample_dataset_mo(name):
    sample_dataset = {
        "catalyst": ["P1-L3"], "t_res": [600], "temperature": [30],"catalyst_loading": [0.498],
    }

    dataset_df = pd.DataFrame(sample_dataset)

    emulator = get_pretrained_reizman_suzuki_emulator(case=1)
    conditions = DataSet.from_df(dataset_df)
    emulator_output = emulator.run_experiments(conditions, rtn_std=True)

    rxn_yield = emulator_output.to_numpy()[0, 5]
    rxn_ton = emulator_output.to_numpy()[0, 4]

    dataset_df['yield'] = rxn_yield *100
    dataset_df['ton'] = rxn_ton *10 
    dataset_df['iteration'] = 0

    variable_df = pd.DataFrame(dataset_df.columns, columns=["variables"])
    sample_data = Data(
        name=f"{name}-sample-reizman-mo", #"sample-reizman-suzuki-mo",
        data=dataset_df.to_json(orient="records"),
        variables=variable_df.to_json(orient="records"),
        fidelity = 'SINGLE',
        user_id=current_user.id,
    )
    db.session.add(sample_data)
    db.session.flush()
    db.session.commit()


def please_add_sample_dataset_mfbo(name):
    sample_dataset = {
         "t_res": [600, 530], "temperature": [30, 60],"catalyst_loading": [0.498, 1.2], "fidelity": [1.0, 0.01],
    }

    dataset_df = pd.DataFrame(sample_dataset)

    emulator = get_pretrained_reizman_suzuki_emulator(case=1)
    yields = []

    for index, row in dataset_df.iterrows():
        # Create a DataFrame for the current row
        condition_df = pd.DataFrame([row])
        # Convert the DataFrame to a DataSet object
        
        condition_df.insert(0, 'catalyst', 'P1-L1')
        conditions = DataSet.from_df(condition_df)

        fidelity_value = row['fidelity']
        emulator_output = emulator.run_experiments(conditions, rtn_std=True)

        rxn_yield = emulator_output.to_numpy()[0, 5]
        if fidelity_value == 1.0:
            yield_value = rxn_yield*10

        elif fidelity_value == 0.01:
            yield_value = rxn_yield + random.gauss(0, 6) #adding gaussian noise to LF data

        yields.append(yield_value) 


    dataset_df['yield'] = yields
    dataset_df['iteration'] = 0
    variable_df = pd.DataFrame(dataset_df.columns, columns=["variables"])
    variable_df = variable_df[variable_df.variables != "fidelity"]
    sample_data = Data(
        name=f"{name}-sample-reizman-mfbo", #"sample-reizman-suzuki-mo",
        data=dataset_df.to_json(orient="records"),
        variables=variable_df.to_json(orient="records"),
        fidelity = 'MULTI',
        fidelity_column = 'fidelity',
        user_id=current_user.id,
    )
    db.session.add(sample_data)
    db.session.flush()
    db.session.commit()



@home_dash.route('/delete-dataset', methods=['POST'])
def delete_dataset():
    data = json.loads(request.data)
    noteId = data['noteId']
    note = Data.query.get(noteId)
    if note:
        if note.user_id == current_user.id:
            db.session.delete(note)
            db.session.commit()
    return jsonify({})


@home_dash.route('/delete-experiment', methods=['POST'])
def delete_experiment():
    data = json.loads(request.data)
    noteId = data['noteId']
    note = Experiment.query.get(noteId)
    if note:
        if note.user_id == current_user.id:
            db.session.delete(note)
            db.session.commit()
    return jsonify({})
