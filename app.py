from flask import Flask, render_template, request, redirect, url_for, session, flash,send_from_directory,make_response,jsonify
from flask_mysqldb import MySQL
from werkzeug.utils import secure_filename
import os
from flask_bcrypt import Bcrypt
from pdfextractor import extract_text_from_pdf
from preprocessor import preprocess
from knn_algo import KNN
from tfidf import TFIDF
from labelencoder import CustomLabelEncoder
from docx import Document
from docxextractor import extract_text_from_docx
from cosineSimilarity import cosine_similarity_sparse
import numpy as np
import pickle
import re
from sklearn.preprocessing import MaxAbsScaler
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
app.config['SECRET_KEY'] = 'XYZ123'
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_PORT'] = 3306
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = '#Haikyuu!!!109'
app.config['MYSQL_DB'] = 'myproject'

mysql = MySQL(app)
bcrypt = Bcrypt(app)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf','docx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Load your trained model
with open('knn_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizerTFIDF.pkl', 'rb') as model_file:
    vectorizer = pickle.load(model_file)

with open('label_encoder.pkl', 'rb') as model_file:
    encoder = pickle.load(model_file)

with open('scalerrr.pkl', 'rb') as model_file:
    scaler = pickle.load(model_file)


# Initialize a dictionary to store cached classification results
# classification_cache = {}

def extract_text_and_classify(file_path, job_id):
    _, file_extension = os.path.splitext(file_path)

        # Extract text from PDF
    if file_extension.lower() == '.pdf':
        text = extract_text_from_pdf(file_path)
        # Extract text from DOCX
    elif file_extension.lower() == '.docx':
        text = extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file format. Only PDF and DOCX files are supported.")  

        # Preprocess the text
    preprocessed_text = preprocess(text)  

        # TF-IDF transformation
    tfidf_matrix = vectorizer.transform([preprocessed_text])

    tfidf_matrix_scaled = scaler.transform(tfidf_matrix)

        # Predict using the trained KNN model
    predicted_category = model.predict(tfidf_matrix_scaled)

    cur = mysql.connection.cursor()
    cur.execute("SELECT title, description, requirements from job_postings where id = %s", (job_id,))
    result = cur.fetchone()
    mysql.connection.commit()
    cur.close()

    if result:
        title, description, requirements = result
        
            # Concatenate the title, description, and requirements
        job_text = f"{title}\n\n{description}\n\n{requirements}"

    preprocessed_job_text = preprocess(job_text)

    job_tfidf = vectorizer.transform([preprocessed_job_text])

        # Simulated cosine similarity score
    similarity_score = cosine_similarity_sparse(tfidf_matrix, job_tfidf)

        # Cache the classification results
    # classification_cache[(file_path, job_id)] = (predicted_category[0], similarity_score[0][0], confidence_score)

    return predicted_category[0], similarity_score[0][0]






@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
# ...
############################################
@app.after_request
def add_no_cache_headers(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/')
def home():
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM job_postings")
        job_postings = cur.fetchall()
        cur.close()

        return render_template('home.html', job_postings=job_postings)

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/admin_panel')
def admin_panel():
    if 'user_id' in session:    
        if 'user_id' in session:
            user_email = session['user_id']
            cur = mysql.connection.cursor()

            # Check if the user is an admin
            result_admin = cur.execute("SELECT * FROM admins WHERE email = %s", [user_email])

            if result_admin > 0:
                return render_template('admin_panel.html')
            
        return redirect(url_for('home'))
    else:
        return redirect(url_for('login'))
    



@app.route('/create_job_posting', methods=['GET', 'POST'])
def create_job_posting():
    form_data = {}
    if 'user_id' in session:
        if request.method == 'POST':
            title = request.form['title']
            description = request.form['description']
            requirements = request.form['requirements']
            salary = request.form['salary']
            company = request.form['company']
            negotiable = request.form.get('negotiable', 'non-negotiable')  # Retrieve selected value or default to 'non-negotiable'

            # Form validation
            if not all([title, description, requirements, salary, company]):
                flash('All fields are required', 'error')
                form_data = {
                    'title': title,
                    'description': description,
                    'requirements': requirements,
                    'salary': salary,
                    'company': company
                }
                return render_template('create_job_posting.html', form_data=form_data)
            elif not salary.isdigit() or int(salary) < 0 or int(salary) < 15000:
                flash('please enter a valid salary', 'error_salary')
                form_data = {
                    'title': title,
                    'description': description,
                    'requirements': requirements,
                    'salary': '',
                    'company': company
                }
                return render_template('create_job_posting.html', form_data=form_data)
            elif not re.match(r"^[a-zA-Z0-9\s]*$", company):
                flash('Company name cannot contain symbols', 'error_company')
                form_data = {
                    'title': title,
                    'description': description,
                    'requirements': requirements,
                    'salary': salary,
                    'company': ''
                }
                return render_template('create_job_posting.html', form_data=form_data)
            else:
                cur = mysql.connection.cursor()
                
                # Check if a job posting with the same title and company already exists
                cur.execute("SELECT COUNT(*) FROM job_postings WHERE title = %s AND company = %s", (title, company))
                if cur.fetchone()[0] > 0:
                    flash('A job with the same title already exists for this company', 'error_title')  

                    cur.close()
                    return render_template('create_job_posting.html', form_data=form_data)

                # Insert the job posting if no duplicate is found
                cur.execute(
                    "INSERT INTO job_postings (title, description, requirements, salary, company, negotiable) VALUES (%s, %s, %s, %s, %s, %s)",
                    (title, description, requirements, salary, company, negotiable))
                mysql.connection.commit()
                cur.close()

                flash('Job posting created successfully', 'success')
                return redirect(url_for('admin_panel'))

        return render_template('create_job_posting.html', form_data=form_data)
    else:
        return redirect(url_for('login'))


@app.route('/view_jobs')
def view_jobs():
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM job_postings")
        job_postings = cur.fetchall()
        cur.close()

        return render_template('view_jobs.html', job_postings=job_postings)
    
#for admin view job
@app.route('/view_job_admin')
def view_job_admin():
    if 'user_id' in session:
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM job_postings")
        job_postings = cur.fetchall()
        cur.close()

        return render_template('view_job_admin.html', job_postings=job_postings)
    else:
        return redirect(url_for('login'))
#admin job details


@app.route('/job_detail/<int:job_id>')
def job_detail(job_id):
    if 'user_id' in session:
        user_email = session['user_id']
        cur = mysql.connection.cursor()

        # Check if the user is an admin
        result_admin = cur.execute("SELECT * FROM admins WHERE email = %s", [user_email])

        if result_admin > 0:
            query = """
                SELECT job_applications.id AS application_id, 
                    job_postings.id AS job_id, 
                    job_postings.title AS job_title, 
                    candidates.full_name, 
                    candidates.email, 
                    resumes.file_path
                FROM job_applications
                JOIN job_postings ON job_applications.job_posting_id = job_postings.id
                JOIN candidates ON job_applications.candidate_id = candidates.id
                JOIN resumes ON job_applications.candidate_id = resumes.candidate_id
                WHERE job_postings.id = %s
                """
            cur.execute(query, (job_id,))
            job_applications = cur.fetchone()
            cur.close()

            return render_template('job_applications.html', job_applications=job_applications)

        return redirect(url_for('home'))
    else:
        return redirect(url_for('login'))

# ...


@app.route('/view_job/<int:job_id>')
def view_job(job_id):
    # if 'user_id' in session:
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM job_postings WHERE id = %s", [job_id])
        job_posting = cur.fetchone()
        cur.close()

        return render_template('view_job.html', job_posting=job_posting)
    # else :
    #     return redirect(url_for('login'))

@app.route('/apply_job/<int:job_id>', methods=['GET', 'POST'])
def apply_job(job_id):
    if 'user_id' in session:
        user_email = session['user_id']
        #####################################################
                # Check if the candidate has already applied for this job posting
        cur = mysql.connection.cursor()
        result = cur.execute(
            "SELECT * FROM job_applications WHERE candidate_id = (SELECT id FROM candidates WHERE email = %s) AND job_posting_id = %s",
            (user_email, job_id)
        )
        
        if result > 0:
            # Candidate has already applied, show a message or redirect as needed
            flash('You have already applied for this job.', 'error')
            return redirect(url_for('view_job', job_id=job_id))

        #################################################

        if request.method == 'POST':
            # Check if the post request has the file part
            if 'resume' not in request.files:
                flash('No file part', 'error')
                return redirect(request.url)

            resume_file = request.files['resume']

            # If the user does not select a file, the browser will submit an empty part without a filename
            if resume_file.filename == '':
                flash('No selected file', 'error')
                return redirect(request.url)

            # Check if the file extension is allowed
            if resume_file and allowed_file(resume_file.filename):
                # Save the resume to the uploads folder
                filename = secure_filename(resume_file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                resume_file.save(file_path)

                # Store the resume information in the database
                cur = mysql.connection.cursor()
                cur.execute(
                    "INSERT INTO resumes (candidate_id, file_path) VALUES ((SELECT id FROM candidates WHERE email = %s), %s)",
                    (user_email, filename))
                mysql.connection.commit()
                cur.close()

                # Fetching the newly inserted resume_id
                cur = mysql.connection.cursor()
                cur.execute("SELECT LAST_INSERT_ID()")
                resume_id = cur.fetchone()[0]
                cur.close()

                #store the information in application table
                cur = mysql.connection.cursor()
                cur.execute(
                    "INSERT INTO job_applications (job_posting_id, candidate_id, resume_id) VALUES (%s, (SELECT id FROM candidates WHERE email = %s), %s)",
                    (job_id, user_email, resume_id))
                mysql.connection.commit()
                cur.close()

                flash('Resume submitted successfully', 'success')
                return redirect(url_for('view_job', job_id=job_id))

        return render_template('apply_job.html', job_id=job_id)

    else:
        flash('Please login to apply for the job', 'error')
         
        return redirect(url_for('view_job', job_id=job_id))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ...

#fetching data for displaying in job_application page----- client data
def get_job_applications_data(job_id):
    connection = mysql.connection
    cursor = connection.cursor()

    query = """
            SELECT
            job_applications.id AS application_id,
            job_applications.resume_id as r_id,
            job_postings.id AS job_id,
            job_postings.title AS job_title,
            candidates.full_name,
            candidates.email,
            resumes.file_path
        FROM
            job_applications
        JOIN job_postings ON job_applications.job_posting_id = job_postings.id
        JOIN candidates ON job_applications.candidate_id = candidates.id
        JOIN resumes ON job_applications.resume_id = resumes.id
        WHERE
            job_applications.job_posting_id = %s;
    """
    
    cursor.execute(query,(job_id,))

    columns = [column[0] for column in cursor.description]
    job_applications_data = [dict(zip(columns, row)) for row in cursor.fetchall()]

    # Classify and store results in the database
    for application in job_applications_data:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], application['file_path'])
        
        # Check if classification results exist in the database
        cursor.execute("""
            SELECT classified_category, similarity_score
            FROM classified_resumes
            WHERE file_path = %s AND job_posting_id = %s
        """, (application['file_path'], application['job_id']))
        existing_record = cursor.fetchone()
        
        if existing_record:
            # Retrieve existing classification results from the database
            predicted_category, similarity_score = existing_record
        else:
            # Compute classification results
            predicted_category, similarity_score = extract_text_and_classify(file_path, application['job_id'])
            predicted_category = encoder.inverse_transform(predicted_category)

            # Insert new record into classified_resumes table
            cursor.execute("""
                INSERT INTO classified_resumes (classified_category, similarity_score, resume_id, job_posting_id, file_path) 
                VALUES (%s, %s, %s, %s, %s)
            """, (predicted_category, similarity_score, application['r_id'], application['job_id'], application['file_path']))
            
        # Update the job_applications_data dictionary with classification results
        application['classified_category'] = predicted_category
        application['similarity_score'] = similarity_score

    connection.commit()
    cursor.close()

    return job_applications_data


def get_classified_resume_data(job_id):
    connection = mysql.connection
    cursor = connection.cursor()

    query = """
        SELECT *
        FROM classified_resumes
        where job_posting_id= %s;
    """
    
    cursor.execute(query,(job_id,))

    columns = [column[0] for column in cursor.description]
    classified_resume_data = [dict(zip(columns, row)) for row in cursor.fetchall()]

    classified_resume_data_sorted = sorted(classified_resume_data, key=lambda x: x['similarity_score'], reverse=True)

    for i, application in enumerate(classified_resume_data_sorted, start=1):
        application['rank'] = i


    cursor.close()

    return classified_resume_data_sorted


@app.route('/classified_resumes<int:job_id>', methods=['GET'])
def classified_resumes(job_id):
    if 'user_id' in session:
        classified_resume_data = get_classified_resume_data(job_id)
        return render_template('classified_resumes.html', classified_resumes=classified_resume_data)
    else:
        return redirect(url_for('login'))


@app.route('/job_applications<int:job_id>', methods=['GET'])
def job_applications(job_id):
    if 'user_id' in session:
        job_applications_data = get_job_applications_data(job_id)
        return render_template('job_applications.html', job_applications=job_applications_data,jobId= job_id)
    else :
        return redirect(url_for('login'))



##############################################################
#registerroute
# Registration route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        full_name = request.form['full_name']
        email = request.form['email']
        password = request.form['password']
        age = request.form['age']
        gender = request.form['gender']
        address = request.form['address']
        register_as = request.form['register_as']

        # Name validation
        if not re.match(r'^[a-zA-Z\s]*$', full_name):
            flash('Name can only contain letters and spaces, and no numbers or symbols', 'error_name')


        # Email validation
        if not re.match(r'^[a-zA-Z0-9._%+-]+@gmail.com$', email):
            flash('Invalid email address format. Please use a valid Gmail address.', 'error_email')

        # Password validation
        if len(password) < 6 or not any(char.isdigit() for char in password) or not any(char.isupper() for char in password):
            flash('Password must be at least 6 characters long with at least 1 capital letter and 1 numerical value.', 'error_password')

        # Age validation
        try:
            age = int(age)
            if age < 18:
                flash('You must be at least 18 years old to register.', 'error_age')
        except ValueError:
            flash('Invalid age format. Please enter a valid age.', 'error_age')

        if '_flashes' in session:
            return render_template('register.html')

        # Check if email already exists in the database
        cur = mysql.connection.cursor()
        cur.execute("SELECT COUNT(*) FROM candidates WHERE email = %s", (email,))
        candidate_result = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM admins WHERE email = %s", (email,))
        admin_result = cur.fetchone()[0]

        if candidate_result > 0 or admin_result > 0:
            flash('An account with this email already exists. Please use a different email address.', 'error_email')
            cur.close()
            return render_template('register.html')

        password_hash = bcrypt.generate_password_hash(password).decode('utf-8')

        if register_as == 'candidate':
            cur.execute(
                "INSERT INTO candidates (full_name, email, password, age, gender, address) VALUES (%s, %s, %s, %s, %s, %s)",
                (full_name, email, password_hash, age, gender, address))
            mysql.connection.commit()
        elif register_as == 'admin':
            cur.execute(
                "INSERT INTO admins (full_name, email, password, age, gender, address) VALUES (%s, %s, %s, %s, %s, %s)",
                (full_name, email, password_hash, age, gender, address))
            mysql.connection.commit()

        cur.close()
        return redirect(url_for('login'))  # Redirect to the login page after registration

    return render_template('register.html')




#login route

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password_candidate = request.form['password']

        cur = mysql.connection.cursor()

        session.clear()

        # Check if the user is an admin
        result_admin = cur.execute("SELECT * FROM admins WHERE email = %s", [email])
        if result_admin > 0:
            data_admin = cur.fetchone()
            password_admin = data_admin[3]
            if bcrypt.check_password_hash(password_admin, password_candidate):
                session['user_id'] = email
                return redirect(url_for('admin_panel'))

        # Check if the user is a candidate
        result_candidate = cur.execute("SELECT * FROM candidates WHERE email = %s", [email])
        if result_candidate > 0:
            data_candidate = cur.fetchone()
            password_candidate_db = data_candidate[3]
            if bcrypt.check_password_hash(password_candidate_db, password_candidate):
                session['user_id'] = email
                return redirect(url_for('home'))

        return render_template('login.html', error='Invalid email or password')

    return render_template('login.html')
###edit job
# ...

@app.route('/edit_job/<int:job_id>', methods=['GET', 'POST'])
def edit_job(job_id):
    if 'user_id' in session:
        user_email = session['user_id']

        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM admins WHERE email = %s", [user_email])

        result_admin = cur.fetchone()

        if result_admin:
            # The user is an admin, proceed with job editing

            if request.method == 'POST':
                title = request.form['title']
                description = request.form['description']
                requirements = request.form['requirements']
                salary = request.form['salary']
                company = request.form['company']  # Add company input

                # Company name validation
                if not re.match(r"^[a-zA-Z0-9\s]*$", company):
                    flash('Company name cannot contain symbols', 'error_company')
                    return redirect(request.url)

                # Salary validation
                # Salary validation
                try:
                    salary = float(salary)
                    if salary < 0 or salary < 15000 :
                        flash('Please enter a valid salary', 'error_salary')
                        return redirect(request.url)
                except ValueError:
                    flash('Invalid salary format. Please enter a valid number.', 'error_salary')
                    return redirect(request.url)



                cur.execute(
                    "UPDATE job_postings SET title=%s, description=%s, requirements=%s, salary=%s, company=%s WHERE id=%s",
                    (title, description, requirements, salary, company, job_id)
                )
                mysql.connection.commit()
                cur.close()

                flash('Job posting updated successfully', 'success')
                return redirect(url_for('view_job_admin'))

            cur.execute("SELECT * FROM job_postings WHERE id = %s", [job_id])
            job_posting = cur.fetchone()
            cur.close()

            return render_template('edit_job.html', job_posting=job_posting)

        else:
            # The user is not an admin, redirect them to the home page
            return redirect(url_for('home'))

    else:
        flash('Please login to edit the job posting', 'error')
        return redirect(url_for('login'))


@app.route('/delete_job/<int:job_id>', methods=['POST'])
def delete_job(job_id):
    if 'user_id' in session:
        user_email = session['user_id']

        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM admins WHERE email = %s", [user_email])

        result_admin = cur.fetchone()

        if result_admin:
            # The user is an admin, proceed with job deletion

            cur.execute("""
                SELECT resumes.file_path
                FROM resumes
                JOIN job_applications ON resumes.id = job_applications.resume_id
                WHERE job_applications.job_posting_id = %s
            """, [job_id])
            resume_files = [row[0] for row in cur.fetchall()]

            # Delete classified resumes associated with the job posting
            cur.execute("DELETE FROM classified_resumes WHERE job_posting_id = %s", [job_id])

            # Delete resumes associated with the job posting
            cur.execute("DELETE FROM resumes WHERE id IN (SELECT resume_id FROM job_applications WHERE job_posting_id = %s)", [job_id])

            # Delete job applications associated with the job posting
            cur.execute("DELETE FROM job_applications WHERE job_posting_id = %s", [job_id])

            # Commit the transaction
            mysql.connection.commit()

            # Delete files from the upload folder
            for file_path in resume_files:
                # Ensure file path is a string
                if isinstance(file_path, str):
                    # Construct the absolute path to the file
                    full_path = os.path.join(app.config['UPLOAD_FOLDER'], file_path)
                    # Check if the file exists and delete it
                    if os.path.exists(full_path):
                        os.remove(full_path)


            # Delete the job posting itself
            cur.execute("DELETE FROM job_postings WHERE id = %s", [job_id])
            mysql.connection.commit()

            cur.close()

            flash('Job posting and associated records deleted successfully', 'success')
            return redirect(url_for('view_job_admin'))

        else:
            # The user is not an admin, redirect them to the home page
            return redirect(url_for('home'))

    else:
        flash('Please login to delete the job posting', 'error')
        return redirect(url_for('login'))



@app.route('/submit_contact', methods=['POST', 'GET'])
def submit_contact():
    name = request.form['name']
    email = request.form['email']
    subject = request.form['subject']
    message = request.form['message']

     # Name validation
    if not re.match(r'^[a-zA-Z\s]*$', name):
        flash('Name can only contain letters and spaces, and no numbers or symbols', 'error_name')
        return redirect(url_for('contact'))

    # Email validation
    if not re.match(r'^[\w\.-]+@[\w\.-]+$', email):
        flash('Please enter a valid email address', 'error_email')
        return redirect(url_for('contact'))

    # Insert data into the database
    # try:
    cur = mysql.connection.cursor()
    cur.execute(
        "INSERT INTO contacts (name, email, subject, message) VALUES (%s, %s, %s, %s)",
        (name, email, subject, message)
    )
    mysql.connection.commit()
    cur.close()
    flash('Your message has been sent successfully', 'success')
    # except Exception as e:
    #     flash('An error occurred while sending the message. Please try again later.', 'error')
    #     print("Error:", e)

    return redirect(url_for('contact'))

@app.route('/admin/messages')
def message_admin():
    # try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM contacts")
        messages = cur.fetchall()
        cur.close()
        return render_template('message_admin.html', messages=messages)
    # except Exception as e:
    #     flash('An error occurred while fetching messages.', 'error')
    #     print("Error:", e)
    #     return redirect(url_for('admin_panel'))

@app.route('/admin/messages/delete/<int:message_id>', methods=['POST'])
def delete_message(message_id):
    
        delete_message_from_database(message_id)  # Implement this function
        flash('Message deleted successfully', 'success')
    
        return redirect(url_for('message_admin'))

def delete_message_from_database(message_id):
    try:
        cur = mysql.connection.cursor()
        cur.execute("DELETE FROM contacts WHERE id = %s", (message_id,))
        mysql.connection.commit()
        cur.close()
    except Exception as e:
        raise e

# ...
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.clear()
    return redirect(url_for('home'))

# main driver function
if __name__ == '__main__':
    app.run(debug=True)
    #http_server = WSGIServer(('', 5000), app)
    #http_server.serve_forever()
    #from waitress import serve
    #serve(app, host="0.0.0.0", port=8080)