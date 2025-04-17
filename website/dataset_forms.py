from .datalab_data import DatalabData
from flask import Blueprint, render_template, request, flash, jsonify, redirect, url_for, session
from flask_login import login_required, current_user
from .models import Data, Experiment
from . import db 
import json
import pandas as pd
from werkzeug.utils import secure_filename
from flask_wtf import FlaskForm
from wtforms import SelectField, DecimalField, SubmitField
from wtforms.validators import DataRequired, NumberRange
# from . import bo_integration

class FidelityForm(FlaskForm):
    fidelity_column = SelectField('Select fidelity column', validators=[DataRequired()])
    #target_fidelity = DecimalField('Target fidelity', validators=[DataRequired(), NumberRange(min=0)], places=3)
    submit = SubmitField('Submit')


data_views = Blueprint("dataset_forms", __name__)


@data_views.route("/select-method", methods=["GET", "POST"])
@login_required
def select_upload_method():
    if request.method == "POST":
        if request.form['action'] == "csv":
            return redirect(url_for("dataset_forms.upload"))
        elif request.form['action'] == "datalab":
            return redirect(url_for("dataset_forms.connect"))
    return render_template("select_dataset_upload_method.html", user=current_user)


@data_views.route("/send/<string:expt_name>", methods=["POST", "GET"])
@login_required
def send(expt_name):
    expt = [row for row in Experiment.query.filter_by(name=expt_name).all()][0]
    df = pd.read_json(expt.data)
    send_df = df[df['iteration'] > 0]

    if request.method == "POST":
        api_key = request.form.get('api_key')
        domain = request.form.get('domain')
        collection_id = request.form.get('collection_id')
        datalab_instance = DatalabData(
            api_key=api_key,
            domain=domain,
            collection_id=collection_id,
        )
        datalab_instance.create_new_samples(send_df)
        flash("your measurements have been sent!", category="success")
        return redirect(url_for('home_dash.view_experiment', expt_name=expt.name))

    return render_template("send_datalab.html", user=current_user)


@data_views.route("/connect", methods=["GET", "POST"])
@login_required
def connect():
    if request.method == "POST":
        if request.form['action'] == "submit":
            # check the dataset name
            name = request.form.get('dataset_name')
            if db.session.query(Data.id).filter_by(name=name).scalar() is not None:
                flash("Dataset names must be unique.", category="error")
            else:
                api_key = request.form.get('api_key')
                domain = request.form.get('domain')
                collection_id = request.form.get('collection_id')
                blocktype = request.form.get('block_id')
                features = request.form.get('parameter_names').split(',')
                features = [feature.strip() for feature in features]

                data = DatalabData(api_key, domain, collection_id, blocktype, features)
                df = data.get_data()
                variable_df = pd.DataFrame(df.columns, columns=["variables"])
                df['iteration'] = 0
                
                input_data = Data(
                    name=f"datalab-{name}",
                    data=df.to_json(orient='records'),
                    variables=variable_df.to_json(orient='records'),
                    user_id=current_user.id
                )
                db.session.add(input_data)
                db.session.flush()
                db.session.commit()
                flash("Upload successful!", category="success")
                return redirect(url_for("home_dash.home"))
    return render_template("connect_datalab.html", user=current_user)


@data_views.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "POST":
        datafile = request.files["formFile"]
        dataset_type = request.form.get("datasetType")
        if datafile:
            filename = secure_filename(datafile.filename)
            datafile.save(filename)
            df = pd.read_csv(filename)
            variable_df = pd.DataFrame(df.columns, columns=["variables"])
            df['iteration'] = 0
            name = request.form.get('dataName')
            if db.session.query(Data.id).filter_by(name=name).scalar() is not None:
                flash("Dataset names must be unique.", category="error")
            else:
                if dataset_type == "single":
                    input_data = Data(
                        name=f"{name}",
                        data=df.to_json(orient='records'),
                        variables=variable_df.to_json(orient='records'),
                        fidelity = 'SINGLE',
                        user_id=current_user.id
                    )
                    print(input_data)
                    db.session.add(input_data)
                    db.session.flush()
                    db.session.commit()
                    # session["dataID"] = input_data.id
                    flash("Upload successful!", category="success")
                    return redirect(url_for("home_dash.home"))
                elif dataset_type == "multi":

                    session['filename'] = filename
                    session['dataName'] = name
                    return redirect(url_for("dataset_forms.multi_fidelity_info"))
        else:
            flash("Upload unsuccessful. Try again.", category="error")
    return render_template("upload.html", user=current_user)


#currently not being accessed
@data_views.route("/multi_fidelity_info", methods=["GET", "POST"])
@login_required
def multi_fidelity_info():
    form = FidelityForm()
    filename = session.get('filename')
    name = session.get('dataName')

    if filename and name:
        df = pd.read_csv(filename)
        form.fidelity_column.choices = [(col, col) for col in df.columns]

        if form.validate_on_submit():
            fidelity_column = form.fidelity_column.data
            variable_df = pd.DataFrame(df.columns, columns=["variables"])
            print(variable_df, 'before')
            variable_df = variable_df[variable_df.variables != fidelity_column]
            print(variable_df, 'after')
            df['iteration'] = 0
            print(fidelity_column)
            # Process the multi-fidelity data as needed
            input_data = Data(
                name=f"{name}",
                data=df.to_json(orient='records'),
                variables=variable_df.to_json(orient='records'),
                fidelity = 'MULTI',
                fidelity_column = fidelity_column,
                user_id=current_user.id
            )
            print(input_data)
            db.session.add(input_data)
            db.session.flush()
            db.session.commit()
            flash("Multi-fidelity dataset upload successful!", category="success")
            return redirect(url_for("home_dash.home"))
    else:
        flash("Session data missing. Please re-upload your dataset.", category="error")
        return redirect(url_for("dataset_forms.upload"))

    return render_template("mf_dataset_info.html", form=form, user=current_user)