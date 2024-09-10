from flask import Flask, redirect, url_for, render_template, request
import os
import cv2
import pandas as pd
from skimage import measure, color, io
from keras.utils import normalize
import matplotlib.pyplot as plt
from UNet_Model import unet_model
import numpy as np
from werkzeug.utils import secure_filename
import sys
from flask_sqlalchemy import SQLAlchemy
import io
from math import ceil
import math
import base64
import matplotlib
matplotlib.use('Agg')
from collections import Counter
from datetime import datetime, timezone, timedelta
import pytz
import plotly.graph_objects as go
from PIL import Image

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf8')

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///liberica_bean_metadata_dummy2.db'
db = SQLAlchemy(app)

local_tz = pytz.timezone('Asia/Manila')

def get_current_timestamp():
    return datetime.now(timezone.utc).astimezone(local_tz).replace(microsecond=0)

def resize_image(filepath, output_path, size=(256, 256)):
    img = Image.open(filepath)
    img = img.resize(size, Image.Resampling.LANCZOS)
    img.save(output_path)

class LibericaBeanMetadata(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filepath = db.Column(db.String)
    image_id = db.Column(db.String)
    area = db.Column(db.Float)
    perimeter = db.Column(db.Float)
    equivalent_diameter = db.Column(db.Float)
    extent = db.Column(db.Float)
    mean_intensity = db.Column(db.Float)
    solidity = db.Column(db.Float)
    convex_area = db.Column(db.Float)
    axis_major_length = db.Column(db.Float)
    axis_minor_length = db.Column(db.Float)
    eccentricity = db.Column(db.Float)
    class_label = db.Column(db.String)
    created_at = db.Column(db.DateTime(timezone=True), default=get_current_timestamp)

app.liberica_bean_metadata = LibericaBeanMetadata

class CoffeeBeanAnalyzer:
    def __init__(self, img_height, img_width, img_channels):
        self.IMG_HEIGHT = img_height
        self.IMG_WIDTH = img_width
        self.IMG_CHANNELS = img_channels
        self.model = self.get_model()
        self.model.load_weights('unet_model/coffee_bean_test-20.hdf5')

    def get_model(self):
        return unet_model(self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS)

    def load_and_process_image(self, filepath):
        img = cv2.imread(filepath, 0)
        img = cv2.resize(img, (self.IMG_WIDTH, self.IMG_HEIGHT))
        img_norm = np.expand_dims(normalize(np.array(img), axis=1), 2)
        img_norm = img_norm[:, :, 0][:, :, None]
        img_input = np.expand_dims(img_norm, 0)
        return img_input

    def segment_image(self, img):
        return (self.model.predict(img)[0, :, :, 0] > 0.9).astype(np.uint8)

    def save_segmented_image(self, segmented, output_filename):
        plt.imsave(output_filename, segmented, cmap='gray')

    def apply_watershed_algorithm(self, img):
        img_grey = img[:, :, 0]
        ret1, thresh = cv2.threshold(img_grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=10)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret2, sure_fg = cv2.threshold(dist_transform, 0.01 * dist_transform.max(), 255, 0)
        sure_fg = np.array(sure_fg, dtype=np.uint8)
        unknown = cv2.subtract(sure_bg, sure_fg, dtype=cv2.CV_32S)
        ret3, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 10
        markers[unknown == 255] = 0
        cv2.watershed(img, markers)
        return markers

    def extract_properties(self, markers, img):
        props = measure.regionprops_table(markers, intensity_image=img[:, :, 0],
                                          properties=['area', 'perimeter', 'equivalent_diameter', 'extent', 'mean_intensity', 'solidity', 'convex_area', 'axis_major_length', 'axis_minor_length', 'eccentricity'])
        return props

    def create_dataframe(self, props, class_label, filepath):
        df = pd.DataFrame(props)
        df = df[df.mean_intensity > 100]
        df['class_label'] = class_label
        image_id = os.path.basename(filepath)
        df= df[['area', 'perimeter', 'equivalent_diameter', 'extent', 'mean_intensity', 'solidity', 'convex_area', 'axis_major_length', 'axis_minor_length', 'eccentricity', 'class_label']]
        df.insert(0, 'filepath', filepath)
        df.insert(1, 'image_id', image_id)
        return df

    def save_to_database(self, df):
        for index, row in df.iterrows():
            metadata = LibericaBeanMetadata(
                filepath=row['filepath'],
                image_id=row['image_id'],
                area=row['area'],
                perimeter=row['perimeter'],
                equivalent_diameter=row['equivalent_diameter'],
                extent=row['extent'],
                mean_intensity=row['mean_intensity'],
                solidity=row['solidity'],
                convex_area=row['convex_area'],
                axis_major_length=row['axis_major_length'],
                axis_minor_length=row['axis_minor_length'],
                eccentricity=row['eccentricity'],
                class_label=row['class_label']
            )
            db.session.add(metadata)
        db.session.commit()

analyzer = CoffeeBeanAnalyzer(img_height=256, img_width=256, img_channels=1)


@app.route('/')
def homepage():
    return render_template("/homepage.html")

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    """Dashboard route handler"""
    data = load_recent_data()
    avg_data = calculate_monthly_averages(data)
    selected_feature, class_label = handle_feature_selection(request, avg_data)
    graph_div = create_graph(avg_data, selected_feature, class_label)
    class_labels = avg_data['class_label'].unique().tolist()
    return render_template("/dashboard.html", 
                           graph_div=graph_div, 
                           features=[col for col in avg_data.columns[1:] if col != 'class_label'], 
                           title=f'Liberica Bean Metadata Dashboard - {selected_feature} - {class_label}'.title(), 
                           selected_feature=selected_feature, 
                           class_label=class_label, 
                           class_labels=class_labels)

def load_recent_data() -> pd.DataFrame:
    """Load recent data from database"""
    five_months_ago = datetime.now() - timedelta(days=150)
    data = LibericaBeanMetadata.query.filter(LibericaBeanMetadata.created_at >= five_months_ago).all()
    return pd.DataFrame([(d.created_at, d.area, d.perimeter, d.equivalent_diameter, d.extent, 
                             d.axis_major_length, d.axis_minor_length, d.eccentricity, d.class_label) 
                             for d in data], 
                           columns=['created_at', 'area', 'perimeter', 'equivalent_diameter', 'extent', 
                                    'axis_major_length', 'axis_minor_length', 'eccentricity', 'class_label'])

def calculate_monthly_averages(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate monthly averages"""
    data['created_at'] = data['created_at'].dt.to_period('M')
    avg_data = data.groupby(['created_at', 'class_label']).mean().reset_index()
    avg_data['created_at'] = avg_data['created_at'].dt.strftime('%Y-%m')
    return avg_data

def handle_feature_selection(request, avg_data: pd.DataFrame) -> tuple:
    """Handle feature selection from request"""
    selected_feature = None
    class_label = None
    if request.method == 'POST':
        if 'feature' in request.form:
            selected_feature = request.form['feature']
        if 'class_label' in request.form:
            class_label = request.form['class_label']
    return selected_feature, class_label

def create_graph(avg_data: pd.DataFrame, selected_feature: str, class_label: str) -> str:
    """Create graph"""
    fig = go.Figure()
    if selected_feature is not None and class_label is not None:
        if selected_feature == 'Area':
            for feature in avg_data.columns[1:]:
                filtered_data = avg_data[avg_data['class_label'] == class_label]
                fig.add_trace(go.Bar(x=filtered_data['created_at'], y=filtered_data[feature], 
                                      name=feature.replace('_', ' ').capitalize()))
        else:
            filtered_data = avg_data[avg_data['class_label'] == class_label]
            fig.add_trace(go.Bar(x=filtered_data['created_at'], y=filtered_data[selected_feature], 
                                  name=selected_feature.replace('_', ' ').capitalize(), marker_color='#591f0b'))
        fig.update_layout(xaxis=dict(tickmode='array', tickvals=avg_data['created_at'], 
                                      ticktext=avg_data['created_at']), 
                          yaxis=dict(visible=False))
    return fig.to_html(full_html=False)

@app.route('/objectives')
def objectives():
    return render_template('objectives.html')

@app.route('/records')
def records():
    """
    Display paginated and sorted records with filtering by class_label
    """
    data = LibericaBeanMetadata.query.all()
    page_size = 100
    page = int(request.args.get('page', 1))  # get the current page from the query string
    sort_by = request.args.get('sort_by', 'id')  # get the column to sort by
    sort_order = request.args.get('sort_order', 'asc')  # get the sort order (asc or desc)
    class_label = request.args.get('class_label')  # get the class label for filtering

    # fetch unique class_label values
    class_labels = set(d.class_label for d in data)

    # filter data by class_label if provided
    if class_label:
        data = [d for d in data if d.class_label == class_label]

    # calculate the total number of pages
    total_pages = math.ceil(len(data) / page_size)

    # calculate the start and end indices for the current page
    start = (page - 1) * page_size
    end = start + page_size

    # slice the data for the current page
    paginated_data = data[start:end]

    # sort the paginated data
    if sort_by and sort_order:
        if sort_order == 'asc':
            paginated_data = sorted(paginated_data, key=lambda x: getattr(x, sort_by))
        else:
            paginated_data = sorted(paginated_data, key=lambda x: getattr(x, sort_by), reverse=True)

    return render_template('records.html', data=paginated_data, page=page, total_pages=total_pages, sort_by=sort_by, sort_order=sort_order, class_labels=class_labels, class_label=class_label)

@app.route('/scan', methods=['GET', 'POST'])
def scan():
    if request.method == 'POST':
        file = request.files['file']
        class_label = request.form['class_label']
        filepath = os.path.join('uploads', file.filename)
        file.save(filepath)

        # Resize the image to 256x256
        output_path = os.path.join('uploads', f'{file.filename}.png')
        resize_image(filepath, output_path)

        analyzer = CoffeeBeanAnalyzer(img_height=256, img_width=256, img_channels=1)
        img = analyzer.load_and_process_image(output_path)
        segmented = analyzer.segment_image(img)
        output_filename = os.path.join('for_watershed', f'{file.filename}')
        analyzer.save_segmented_image(segmented, output_filename)
        img = cv2.imread(output_filename)
        markers = analyzer.apply_watershed_algorithm(img)
        props = analyzer.extract_properties(markers, img)
        df = analyzer.create_dataframe(props, class_label, output_path)

        # Save to database
        analyzer.save_to_database(df)

        return render_template('results.html', df=df)
    return render_template('scan_coffee_bean.html')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)