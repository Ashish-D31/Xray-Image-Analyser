<h1>Running the app</h1>
The app uses pretrained models to classify images so the app can be used directly.<br> 
The XrayOrNot.tflite model determines whether the image is Xray or not.<br>
The FracOrNot.tflite model determines whether the bone in the Xray is fractured or not.

<p></p>

## Steps to run the app ##
1. Install all required libraries from requirements.txt
2. Navigate to the app's parent directory and open a terminal in it 
3. Run the command "streamlit run app.py"
4. The site will open on "localhost:8501"
5. Insert the image you want to be analysed and the results will be displayed
