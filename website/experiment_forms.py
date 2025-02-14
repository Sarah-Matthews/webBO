from flask import Blueprint, render_template, request, flash, jsonify, redirect, url_for, session, Flask
from flask_login import login_required, current_user
from .models import Data, Experiment
from . import db 
import json
import pandas as pd
from werkzeug.utils import secure_filename
import werkzeug
# from . import bo_integration
from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, FormField, HiddenField, SubmitField, IntegerField
from wtforms.validators import DataRequired, InputRequired
from .bo_integration import run_bo, rerun_bo
from summit.benchmarks import get_pretrained_reizman_suzuki_emulator
from summit.utils.dataset import DataSet

import plotly.express as px
import plotly.graph_objects as go
import plotly


expt_views = Blueprint("experiment_forms", __name__)

class TargetOptimisationForm(FlaskForm):
    target_name = StringField('Target Name', render_kw={"readonly": True})  # Display the target name
    opt_type = SelectField('Optimisation type', choices=[('maximize', 'Maximise'), ('minimize', 'Minimise')])
   

class DatasetSelectionForm(FlaskForm):
    form_name = HiddenField("form_name")
    name = StringField('experiment name', validators=[DataRequired()], id='experiment_name', render_kw={"placeholder": "Enter your experiment name here"})
    dataset = SelectField('dataset', coerce=str, validators=[DataRequired()], id='dataset_name')
    target = SelectField('Target (i.e. what you want to optimise)', coerce=str, validators=[DataRequired()], id='target_name')
    submit = SubmitField('Submit dataset')

class DatasetSelectionFormMO(FlaskForm):
    form_name = HiddenField("form_name")
    name = StringField('experiment name', validators=[DataRequired()], id='experiment_name', render_kw={"placeholder": "Enter your experiment name here"})
    dataset = SelectField('dataset', coerce=str, validators=[DataRequired()], id='dataset_name')
    #target = SelectMultipleField('Target(s) (i.e. what you want to optimise)', coerce=str, validators=[DataRequired()], id='target_name')  # Multi-selection
    target1 = SelectField('Target 1', coerce=str, validators=[DataRequired()])
    target2 = SelectField('Target 2', coerce=str, validators=[DataRequired()])
    submit = SubmitField('Submit dataset')


#class ParameterSpaceForm(FlaskForm):
#    variable = SelectField('variable', coerce=str, validators=[InputRequired()])


class HyperparameterForm(FlaskForm):
    kernel = SelectField('GP kernel type', id='kernel')
    acqFunc = SelectField('Acquisition Function type', id='acqFunc')
    batch_size = IntegerField('Batch size')
    opt_type = SelectField('Optimization type')
    submit = SubmitField('Submit hyperparameters')

class HyperparameterFormMO(FlaskForm):
    kernel = SelectField('GP kernel type', id='kernel')
    acqFunc = SelectField('Acquisition Function type', id='acqFunc')
    batch_size = IntegerField('Batch size')
    target1_optimisation = SelectField('Optimization type')
    target2_optimisation = SelectField('Optimization type')
    submit = SubmitField('Submit hyperparameters')


class ExperimentSetupForm(FlaskForm):
    dataset_form = FormField(DatasetSelectionForm)
    hyperparamter_form = FormField(HyperparameterForm)
    submit = SubmitField('Run experiment')

class ObjectiveTypeForm(FlaskForm):
    objective_type = SelectField(
        'Select optimisation type',
        choices=[('single', 'Single Objective'), ('multi', 'Multi Objective')],
        id='objective_type'
    )
    submit = SubmitField('Proceed')


@expt_views.route("/setup", methods=["GET", "POST"])
@login_required
def setup():
    dataset_choices = []
    measurement_choices = {}
    for data in current_user.datas:
        dataset_choices.append(data.name)
        measurement_choices[data.name] = list(pd.read_json(data.variables)['variables'])
    print(measurement_choices)
    data_form = DatasetSelectionForm(form_name="data_select")
    data_form.dataset.choices = [(row.id, row.name) for row in Data.query.filter_by(user_id=current_user.id)]
    data_form.target.choices = [(row.id, row.variables) for row in Data.query.filter_by(user_id=current_user.id)]
    expt_names = [row.name for row in Experiment.query.filter_by(user_id=current_user.id)]
    if data_form.name.data in expt_names:
        flash("That name already exists!", category="error")

    hyp_form = HyperparameterForm()
    hyp_form.kernel.choices = ['Matern', 'Tanimoto']
    hyp_form.acqFunc.choices = ['Expected Improvement', 'Probability of Improvement']
    hyp_form.opt_type.choices = ['maximise', 'minimise']
    opt_type_mapping = {"maximise": "MAX", "minimise": "MIN"}
    if request.method == "POST":
        # if request.form.get('expt_btn') == "run-expt":
        if 'expt_btn' in request.form:
            dataset_info = [row for row in Data.query.filter_by(id=data_form.dataset.data).all()]
            target = data_form.target.data
            variable_types = {}
            for index, variable in pd.read_json(dataset_info[0].variables).iterrows():
                col = variable['variables']

                if request.form.get(f"parameterspace-{col}") == "cat":
                    datafile = request.files[f"formFile-{col}"]
                    if datafile:
                        filename = secure_filename(datafile.filename)
                        datafile.save(filename)
                        df = pd.read_csv(filename)
                    variable_types[f"{col}"] = {
                        "parameter-type": request.form.get(f"parameterspace-{col}"),
                        "json": df.to_json(orient="records"),
                    }
                elif request.form.get(f"parameterspace-{col}") == "subs":
                    datafile = request.files.get(f"formFile-{col}")
                    if datafile:
                        filename = secure_filename(datafile.filename)
                        datafile.save(filename)
                        df = pd.read_csv(filename, index_col=0)
                    print(request.form.get(f"exampleRadios-{col}"))
                    print(df.to_dict())
                    variable_types[f"{col}"] = {
                        "parameter-type": request.form.get(f"parameterspace-{col}"),
                        "json": df.to_dict()['smiles'],
                        "encoding": request.form.get(f"exampleRadios-{col}"),
                    }
                elif request.form.get(f"parameterspace-{col}") == "int":
                    if int(request.form.get(f"min-vals-{col}")) < int(request.form.get(f"max-vals-{col}")):
                        variable_types[f"{col}"] = {
                            "parameter-type": request.form.get(f"parameterspace-{col}"),
                            "min": int(request.form.get(f'min-vals-{col}')),
                            "max": int(request.form.get(f"max-vals-{col}")),
                        }
                    else:
                        flash('Min values MUST be less than max values.', category="error")
                elif request.form.get(f"parameterspace-{col}") == "cont":
                    if float(request.form.get(f"min-vals-{col}")) < float(request.form.get(f"max-vals-{col}")):
                        variable_types[f"{col}"] = {
                            "parameter-type": request.form.get(f"parameterspace-{col}"),
                            "min": float(request.form.get(f'min-vals-{col}')),
                            "max": float(request.form.get(f"max-vals-{col}")),
                        }
                    else:
                        flash('Min values MUST be less than max values.', category="error")
            print(variable_types)
            print('target value:',target)
            expt_info = Experiment(
                name=data_form.name.data,
                dataset_name=dataset_info[0].name,
                data=dataset_info[0].data,
                objective = 'SINGLE',
                fidelity = 'SINGLE',
                target=target,
                variables=json.dumps(variable_types),
                kernel=hyp_form.kernel.data,
                acqFunc=hyp_form.acqFunc.data,
                opt_type=opt_type_mapping[hyp_form.opt_type.data],
                batch_size=hyp_form.batch_size.data,
                next_recs=pd.DataFrame().to_json(orient='records'),
                iterations_completed=0,
                user_id=current_user.id
            )
            db.session.add(expt_info)
            db.session.flush()
            db.session.commit()
            flash("Upload successful!", category="success")
            return redirect(url_for('home_dash.view_experiment', expt_name=expt_info.name)) # redirect(url_for('experiment_forms.run_expt', expt_name=expt_info.name))

    return render_template(
        "setup_experiment.html",
        user=current_user,
        data_form=data_form,
        hyp_form=hyp_form,
        variable_names=_get_variable_names(),
    )



@expt_views.route("/setup_mo", methods=["GET", "POST"])
@login_required
def setup_mo():
    dataset_choices = []
    measurement_choices = {}
    for data in current_user.datas:
        dataset_choices.append(data.name)
        measurement_choices[data.name] = list(pd.read_json(data.variables)['variables'])

    data_form = DatasetSelectionFormMO(form_name="data_select")
    data_form.dataset.choices = [(row.id, row.name) for row in Data.query.filter_by(user_id=current_user.id)]
 
    target_options = []
    selected_dataset = data_form.dataset.data
    #for row in Data.query.filter_by(user_id=current_user.id, id = data_form.dataset.data):
        #try:
            #df = pd.read_json(row.variables)  
            #variables = df.get("variables", []).tolist()  
            #target_options.extend([(f"{row.id}-{var}", var) for var in variables])

       # except ValueError as e:
            #print(f"JSON error for row {row.id}: {e}") 
    if selected_dataset:  # Only filter if a dataset is selected
        dataset_info = Data.query.filter_by(id=selected_dataset, user_id=current_user.id).first()
        if dataset_info:
            try:
                df = pd.read_json(dataset_info.variables)  
                variables = df.get("variables", []).tolist()  
                target_options = [(idx, var) for idx, var in enumerate(variables)]  # Store indices!
            except ValueError as e:
                print(f"JSON error for dataset {dataset_info.id}: {e}")

    data_form.target1.choices = target_options
    data_form.target2.choices = target_options

    expt_names = [row.name for row in Experiment.query.filter_by(user_id=current_user.id)]
    if data_form.name.data in expt_names:
        flash("That name already exists!", category="error")

    hyp_form = HyperparameterFormMO()
    hyp_form.kernel.choices = ['Matern', 'Tanimoto']
    hyp_form.acqFunc.choices = ['Expected Improvement', 'Probability of Improvement']
    hyp_form.target1_optimisation.choices = ['maximise', 'minimise']
    hyp_form.target2_optimisation.choices = ['maximise', 'minimise']
    opt_type_mapping = {"maximise": "MAX", "minimise": "MIN"}
    if request.method == "POST":
        # if request.form.get('expt_btn') == "run-expt":
        if 'expt_btn' in request.form:
            dataset_info = [row for row in Data.query.filter_by(id=data_form.dataset.data).all()]
            target1 = data_form.target1.data
            target2 = data_form.target2.data
            print('target one:', target1)
            print('target two:', target2)

            target1 = int(target1)  # Convert to int
            target2 = int(target2) if target2 else None  # Convert to int if selected

            variable_types = {}
            for index, variable in pd.read_json(dataset_info[0].variables).iterrows():
                col = variable['variables']

                if request.form.get(f"parameterspace-{col}") == "cat":
                    datafile = request.files[f"formFile-{col}"]
                    if datafile:
                        filename = secure_filename(datafile.filename)
                        datafile.save(filename)
                        df = pd.read_csv(filename)
                    variable_types[f"{col}"] = {
                        "parameter-type": request.form.get(f"parameterspace-{col}"),
                        "json": df.to_json(orient="records"),
                    }
                elif request.form.get(f"parameterspace-{col}") == "subs":
                    datafile = request.files.get(f"formFile-{col}")
                    if datafile:
                        filename = secure_filename(datafile.filename)
                        datafile.save(filename)
                        df = pd.read_csv(filename, index_col=0)
                    print(request.form.get(f"exampleRadios-{col}"))
                    print(df.to_dict())
                    variable_types[f"{col}"] = {
                        "parameter-type": request.form.get(f"parameterspace-{col}"),
                        "json": df.to_dict()['smiles'],
                        "encoding": request.form.get(f"exampleRadios-{col}"),
                    }
                elif request.form.get(f"parameterspace-{col}") == "int":
                    if int(request.form.get(f"min-vals-{col}")) < int(request.form.get(f"max-vals-{col}")):
                        variable_types[f"{col}"] = {
                            "parameter-type": request.form.get(f"parameterspace-{col}"),
                            "min": int(request.form.get(f'min-vals-{col}')),
                            "max": int(request.form.get(f"max-vals-{col}")),
                        }
                    else:
                        flash('Min values MUST be less than max values.', category="error")
                elif request.form.get(f"parameterspace-{col}") == "cont":
                    if float(request.form.get(f"min-vals-{col}")) < float(request.form.get(f"max-vals-{col}")):
                        variable_types[f"{col}"] = {
                            "parameter-type": request.form.get(f"parameterspace-{col}"),
                            "min": float(request.form.get(f'min-vals-{col}')),
                            "max": float(request.form.get(f"max-vals-{col}")),
                        }
                    else:
                        flash('Min values MUST be less than max values.', category="error")
            print(variable_types)
            expt_info = Experiment(
                name=data_form.name.data,
                dataset_name=dataset_info[0].name,
                objective = 'MULTI',
                fidelity = 'SINGLE',
                data=dataset_info[0].data,
                target=target1,
                target_2=target2,
                variables=json.dumps(variable_types),
                kernel=hyp_form.kernel.data,
                acqFunc=hyp_form.acqFunc.data,
                opt_type = opt_type_mapping[hyp_form.target1_optimisation.data],
                opt_type_2 = opt_type_mapping[hyp_form.target2_optimisation.data],
                batch_size=hyp_form.batch_size.data,
                next_recs=pd.DataFrame().to_json(orient='records'),
                iterations_completed=0,
                user_id=current_user.id,
            )
            db.session.add(expt_info)
            db.session.flush()
            db.session.commit()
            flash("Upload successful!", category="success")
            return redirect(url_for('home_dash.view_experiment', expt_name=expt_info.name)) # need to change this to view_experiment_mo

    return render_template(
        "setup_experiment_mo.html",
        user=current_user,
        data_form=data_form,
        hyp_form=hyp_form,
        variable_names=_get_variable_names(),
        
    )


@expt_views.route("/add_measurements/<string:expt_name>", methods=["GET", "POST"])
@login_required
def add_measurements(expt_name):
    # Load your DataFrame (df) and other relevant data
    df = [pd.read_json(row.data) for row in Experiment.query.filter_by(name=expt_name).all()][0]
    expt_info = [row for row in Experiment.query.filter_by(name=expt_name).all()][0]
    data_info = Data.query.filter_by(name=expt_info.dataset_name).first()
    variable_list = list(df.columns)
    target_column_name = variable_list[int(expt_info.target)]
    if expt_info.target_2 is not None:
        target_2_column_name = variable_list[int(expt_info.target_2)]
    else:
        target_2_column_name = None
    
    recs = pd.read_json(expt_info.next_recs)
    print(recs)
    if len(recs.columns) < 1:
        recs = pd.DataFrame(columns=df.columns)
        recs.loc[0] = 'insert'

    if "sample-reizman" in expt_info.name:
        emulator_status = True
        df4em = recs
        data = df4em.drop(['yield', 'iteration'], axis=1)
        emulator = get_pretrained_reizman_suzuki_emulator(case=1)
        conditions = DataSet.from_df(data)
        emulator_output = emulator.run_experiments(conditions, rtn_std=True)
        print(emulator_output)
        emulator_value = emulator_output.to_numpy()[0, 5]
        emulator_value_2 = emulator_output.to_numpy()[0, 4]

    else:
        emulator_status = False
        emulator_value = None
        emulator_value_2 = None

    if request.method == "POST":
        if request.form['action'] == 'submit_measurements':
            for index, row in recs.iterrows():
                new_measurement = {}
                # concatenate df with the input values from the form
                for column in recs.columns:
                    if column == 'iteration':
                        new_measurement[column] = recs['iteration'].max()
                    else:
                        new_measurement[column] = request.form.get(f"{column}")

                # updte the data entry in the Data DB
                ndf = pd.DataFrame([new_measurement])
                df = pd.concat([df, ndf])

            expt_info.data = df.to_json(orient='records')
            db.session.add(expt_info)
            db.session.flush()
            db.session.commit()

            return redirect(url_for('home_dash.view_experiment', expt_name=expt_info.name))

    return render_template(
        "add_measurements.html",
        user=current_user,
        df=df,
        titles=df.columns.values,
        target_name=target_column_name,
        target_2 = target_2_column_name,
        recs=recs,
        emulator=emulator_status,
        emulator_value=emulator_value,
        emulator_value_2= emulator_value_2,
    )


@expt_views.route("/run_expt/<string:expt_name>", methods=["GET", "POST"])
@login_required
def run_expt(expt_name):
    expt_info = Experiment.query.filter_by(name=expt_name).first()
    data = pd.read_json(expt_info.data)

    recs = pd.read_json(expt_info.next_recs)
    print('recs:',recs)
    variable_list = list(data.columns)
    target_column_name = variable_list[int(expt_info.target)]

    fig = go.Figure([
        go.Scatter(x=list(data['iteration']), y=list(data[list(data.columns)[int(expt_info.target)]]), mode = 'markers', name='Experiment(s) run by user'), # EM: adding -- mode = 'markers' -- means only the data points are shown, no lines connecting them
        go.Scatter(x=list(recs['iteration']), y=list(recs[list(recs.columns)[int(expt_info.target)]]), mode = 'markers', name='Expected outcome of recommended experiment(s)'),
    ])
    fig.update_layout(
        xaxis_title="iteration",
        yaxis_title=f"{target_column_name}",
        legend_title="Key",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="RebeccaPurple"
        )
    )

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Check if Secondary Target Exists i.e. mobo case**
    graphJSON2 = None  
    target_2_column_name = None

    if expt_info.target_2 is not None:
        try:
            target_2_column_name = variable_list[int(expt_info.target_2)]
            fig2 = go.Figure([
                go.Scatter(
                    x=list(data['iteration']), 
                    y=list(data[target_2_column_name]), 
                    mode='markers', 
                    name='Experiment(s) run by user'
                ),
                go.Scatter(
                    x=list(recs['iteration']), 
                    y=list(recs[target_2_column_name]), 
                    mode='markers', 
                    name='Expected outcome of recommended experiment(s)'
                ),
            ])
            fig2.update_layout(
                xaxis_title="iteration",
                yaxis_title=f"{target_2_column_name}",
                legend_title="Key",
                font=dict(
                    family="Courier New, monospace",
                    size=18,
                    color="RebeccaPurple"
                )
            )
            graphJSON2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
        except (IndexError, ValueError):
            print("Invalid target_2 index, skipping second graph.")


    if request.method == "POST":
        if request.form['action'] == 'add':
            return redirect(url_for("experiment_forms.add_measurements", expt_name=expt_info.name))

    #return render_template(
        #"run_expt.html", # beneath this is everyting we need to add to html to make it run properly
        #user=current_user,
        #df=recs.drop(target_column_name, axis=1),
        #df_2=recs.drop(target_2_column_name, axis=1),
        #titles=recs.drop(target_column_name, axis=1).columns.values,
        #titles_2=recs.drop(target_2_column_name, axis=1).columns.values,
        #graphJSON=graphJSON,
        #graphJSON2 = graphJSON2,
        #target=list(data.columns)[int(expt_info.target)], 
        #target_2=list(data.columns)[int(expt_info.target_2)],     )

    return render_template(
    "run_expt.html", # beneath this is everyting we need to add to html to make it run properly
    user=current_user,
    df=recs,  
    titles=recs.columns.values,  
    graphJSON=graphJSON,
    graphJSON2=graphJSON2,
    target=target_column_name, 
    target_2=target_2_column_name if target_2_column_name else None,  
)

#note - edited function to accommodate multiple
def _get_variable_names():
    rows = Data.query.all()
    datas = []
    for row in rows:
        variables = list(pd.read_json(row.variables)['variables'])
        target_id = 0
        for variable in variables:
            datas.append((row.id, target_id, variable))
            target_id += 1
    return datas

@expt_views.route('/_get_expt_info', methods=["POST"])
def _get_expt_info():
    data = request.get_json()
    target_name = data['target']
    dataset_name = data['dataset']
    rows = Data.query.filter_by(id=dataset_name).all()
    datas = []
    for row in rows:
        variables = list(pd.read_json(row.variables)['variables'])
        target_id = 0
        for variable in variables:
            datas.append((target_id, variable))
            target_id += 1
    return jsonify(datas)


@expt_views.route('/_get_dataset_info/')
def _get_dataset_info():
    data_name = request.values.get('dataset', '01')
    rows = Data.query.filter_by(id=data_name).all()
    datas = []
    for row in rows:
        variables = list(pd.read_json(row.variables)['variables'])
        target_id = 0
        for variable in variables:
            datas.append((target_id, variable))
            target_id += 1
    return jsonify(datas)
