from flask import Flask, request, render_template
from werkzeug import secure_filename
import os
import cv2
from image_out import out


upload = os.path.basename('uploads')
extension = ['png','jpg','jpeg']

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = upload

def allowed_extension(fname):
	# return fname
	fname = fname.split(".")
	if len(fname)==1:
		return False
	else:
		if fname[-1].lower() in extension:
			return True
		else:
			return False

@app.route('/', methods=['POST','GET'])
def index():
	return render_template('index.html')


@app.route('/upload', methods=['POST','GET'])
def upload():
	if request.method == 'POST':
		img = request.files['file']
		# print img.filename
		# return allowed_extension(img.filename)
		if img and allowed_extension(img.filename):
			img_name = secure_filename(img.filename)
			path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
			img.save(path)
			result, pred = out(path)
			face, modi, kejru = "No","No","No"
			if pred[0]!=0:
				face = "Yes"
				if pred[1]==1:
					modi = "Yes"
				if pred[2]==1:
					kejru = "Yes"
			# return result
			result = "./"+result
			return render_template("output.html", output_image = result, face= face, modi= modi, kejru= kejru)
			# return "file uploaded"

if __name__=="__main__":
	app.run(debug=True)



